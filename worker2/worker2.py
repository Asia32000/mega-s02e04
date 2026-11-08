"""Job 2 worker (voice-to-memo, skill version 1). Runs on a GitHub Actions runner.

Modes (first word of mode.txt):
  probe   read-only look: the job's tools and skill, a one-word test of the
          AI models, and the clip's transcript. Hands nothing in.
  run     skill steps 1-3: transcribe, one memo_write per finding, memo_search
          to confirm. Stops before submit and saves its state.
  submit  skill step 4: submit the confirmed notes, one per line.

Secrets come from the environment: JOB2_URL (the job's MCP link) and
GEMINI_API_KEY. Everything the harness needs to read is written to ./out,
which the workflow encrypts with the harness's public key before it leaves
the runner. Stdout stays short because Actions logs of a public repo are public.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mcpclient import MCP, _ssl_context  # noqa: E402

SKILL_VERSION = "1"
CLIP = "https://hq.mega.dev/media/two-places/dispatch-2-c14d2d05.mp3"
OUT = os.path.abspath("out")
STATE_IN = os.path.abspath("prev/state.json")   # decrypted by the workflow when present
STATE_OUT = os.path.join(OUT, "state.json")      # re-encrypted by the workflow

# Model choice: tested from this runner in probe mode (one-word prompt, timing).
MODEL_CANDIDATES = ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash",
                    "gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-flash-lite-latest", "gemini-flash-latest"]
MODEL_TIMEOUT = 60
CTX = _ssl_context()


def save(name, obj):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w") as f:
        if isinstance(obj, str):
            f.write(obj)
        else:
            json.dump(obj, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------- Gemini
def gemini(model, prompt, json_mode=False, timeout=MODEL_TIMEOUT):
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0}}
    if json_mode:
        body["generationConfig"]["responseMimeType"] = "application/json"
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "x-goog-api-key": os.environ["GEMINI_API_KEY"]},
        method="POST")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
        data = json.loads(r.read())
    text = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
    return text, round(time.time() - t0, 2)


def list_models():
    req = urllib.request.Request("https://generativelanguage.googleapis.com/v1beta/models?pageSize=200",
                                 headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]})
    with urllib.request.urlopen(req, timeout=30, context=CTX) as r:
        return [m["name"].split("/", 1)[1] for m in json.loads(r.read()).get("models", [])]


def test_models():
    report = {"available_flash": [], "tests": []}
    try:
        avail = list_models()
        report["available_flash"] = [m for m in avail if "flash" in m]
    except Exception as e:
        report["list_error"] = str(e)[:300]
        avail = MODEL_CANDIDATES
    for m in MODEL_CANDIDATES:
        if m not in avail:
            report["tests"].append({"model": m, "skipped": "not listed"})
            continue
        try:
            text, secs = gemini(m, "Reply with one word: ok", timeout=30)
            report["tests"].append({"model": m, "reply": text.strip()[:40], "seconds": secs})
        except urllib.error.HTTPError as e:
            report["tests"].append({"model": m, "error": f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}"})
        except Exception as e:
            report["tests"].append({"model": m, "error": str(e)[:300]})
    return report


def pick_models(cfg_path=os.path.join(HERE, "models.json")):
    with open(cfg_path) as f:
        cfg = json.load(f)
    return cfg["primary"], cfg["fallback"]


# ---------------------------------------------------------------- helpers
def find_key(obj, names):
    if isinstance(obj, dict):
        for n in names:
            if n in obj and isinstance(obj[n], str):
                return obj[n]
        for v in obj.values():
            r = find_key(v, names)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_key(v, names)
            if r is not None:
                return r
    return None


def step_of(data):
    return find_key(data, ["step", "stepToken", "step_token", "nextStep"])


def mmss(v):
    if isinstance(v, (int, float)):
        s = int(v)
        return f"{s // 60}:{s % 60:02d}"
    return str(v).lstrip("@")


def segments_of(tr):
    """Return [(offset 'm:ss', text)] from the transcribe reply, whatever its key names."""
    segs = None
    if isinstance(tr, dict):
        for k in ("segments", "chunks", "lines", "transcript"):
            if isinstance(tr.get(k), list):
                segs = tr[k]
                break
    elif isinstance(tr, list):
        segs = tr
    out = []
    for s in segs or []:
        if not isinstance(s, dict):
            continue
        off = next((s[k] for k in ("offset", "start", "at", "t", "time", "startSeconds") if k in s), None)
        text = next((s[k] for k in ("text", "said", "words", "content") if k in s), None)
        if off is not None and text:
            out.append((mmss(off), str(text).strip()))
    return out


PROMPT = """You turn a tram desk's end-of-shift voice memo into findings.

A finding is anything that happened on a line: a fault, a freeze or stall, a hold, a reset.
List EVERY time one happened, even when the same kind of event repeats.
The talk around it (greetings, opinions, plans, weather, sign-off) is not a finding.

Each transcript segment starts with its offset in the clip as @m:ss.
For each finding give:
  "what": one short sentence of what happened, naming the line, and keeping any
          clock time spoken in the clip exactly as it is said (same words or digits, same clock name);
  "at":   the @m:ss offset of the segment where it is said, copied exactly, without the @.
One finding per item. If one segment holds two findings, make two items with the same "at".

Return only JSON: {"findings": [{"what": "...", "at": "m:ss"}, ...]}

Transcript:
"""


def extract_findings(segs):
    primary, fallback = pick_models()
    transcript = "\n".join(f"@{o} {t}" for o, t in segs)
    last_err = None
    for model in (primary, fallback):
        try:
            text, secs = gemini(model, PROMPT + transcript, json_mode=True)
            data = json.loads(text)
            items = data["findings"] if isinstance(data, dict) else data
            valid = {o for o, _ in segs}
            cleaned = []
            for it in items:
                at = mmss(it["at"])
                if at not in valid:
                    raise ValueError(f"offset {at} is not a segment offset")
                cleaned.append({"what": it["what"].strip().rstrip("."), "at": at})
            return cleaned, {"model": model, "seconds": secs}
        except Exception as e:
            last_err = f"{model}: {e}"
    raise RuntimeError(f"both models failed: {last_err}")


# ---------------------------------------------------------------- modes
def transcribe(m):
    """Call transcribe with the clip, naming the argument as the tool's schema does."""
    tools = m.request("tools/list", {}).get("tools", [])
    schema = next((t.get("inputSchema", {}) for t in tools if t["name"] == "transcribe"), {})
    props = list((schema.get("properties") or {}).keys())
    req = schema.get("required") or props
    arg = next((k for k in req if "url" in k.lower() or "clip" in k.lower() or "audio" in k.lower()), req[0] if req else "url")
    return m.call("transcribe", {arg: CLIP})


def probe(m):
    rep = {}
    tools = m.request("tools/list", {})
    rep["tools"] = [{"name": t["name"], "schema": t.get("inputSchema")} for t in tools.get("tools", [])]
    try:
        sk = m.request("resources/read", {"uri": "skill://voice-to-memo/SKILL.md"})
        rep["skill"] = sk["contents"][0].get("text", "")
    except Exception as e:
        rep["skill_error"] = str(e)
    rep["models"] = test_models()
    save("report.json", rep)
    tr, raw = transcribe(m)
    rep["transcribe_raw"] = raw
    rep["segments"] = segments_of(tr)
    save("report.json", rep)
    print("probe: tools", [t["name"] for t in rep["tools"]], "| segments", len(rep["segments"]))
    for t in rep["models"]["tests"]:
        print("model test:", t.get("model"), t.get("seconds", t.get("skipped", "error")))


def models_only():
    rep = {"models": test_models()}
    save("report.json", rep)
    for t in rep["models"]["tests"]:
        print("model test:", t.get("model"), t.get("seconds", t.get("skipped", "error")))


def run(m):
    rep = {}
    tr, _ = transcribe(m)
    step = step_of(tr)
    segs = segments_of(tr)
    rep["segments"] = segs
    if not segs:
        save("report.json", rep)
        sys.exit("no segments in the transcript")
    findings, rep["model"] = extract_findings(segs)
    notes = [f"{f['what']} @{f['at']}" for f in findings]
    rep["notes_planned"] = notes
    written = []
    for n in notes:
        data, _ = m.call("memo_write", {"text": n, "step": step})
        step = step_of(data) or step
        written.append(n)
    # confirm: search with no query, compare, write what is missing, search again
    for attempt in range(2):
        data, raw = m.call("memo_search", {"step": step})
        step = step_of(data) or step
        rep[f"search_{attempt}"] = raw
        blob = json.dumps(data, ensure_ascii=False)
        missing = [n for n in notes if n not in blob and json.dumps(n, ensure_ascii=False)[1:-1] not in blob]
        rep[f"missing_{attempt}"] = missing
        if not missing:
            break
        for n in missing:
            data, _ = m.call("memo_write", {"text": n, "step": step})
            step = step_of(data) or step
    rep["notes"] = notes
    save("report.json", rep)
    save("state.json", {"step": step, "notes": notes})
    print("run: notes", len(notes), "| still missing after confirm:", len(rep.get("missing_1", rep.get("missing_0", []))))


def submit(m):
    with open(STATE_IN) as f:
        st = json.load(f)
    data, raw = m.call("submit", {"result": "\n".join(st["notes"]), "skillVersion": SKILL_VERSION, "step": st["step"]})
    save("result.json", raw)
    print("submit: status", find_key(data, ["status"]))


def main():
    mode = (sys.argv[1] if len(sys.argv) > 1 else "probe").strip()
    os.makedirs(OUT, exist_ok=True)
    if mode == "models":
        models_only()
        return
    m = MCP(os.environ["JOB2_URL"], os.path.join(OUT, "log.jsonl"), client_name="job2-actions", timeout=180)
    m.initialize()
    try:
        {"probe": probe, "run": run, "submit": submit}[mode](m)
    except Exception as e:
        save("error.txt", repr(e))
        print("worker error (details in the encrypted output)")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
