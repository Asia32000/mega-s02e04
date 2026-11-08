"""Minimal MCP client (Streamable HTTP, JSON-RPC) using only the Python standard library.

Every request and reply is appended to a JSON-lines log, with the job link and
step tokens masked.
"""
import json
import re
import ssl
import time
import urllib.error
import urllib.request


def _ssl_context():
    try:
        import certifi  # present in anaconda and many installs
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


TOKEN_KEYS = re.compile(r"(step|token|receipt|ticket)", re.I)


def mask(obj):
    """Return a copy with long secret-looking values shortened."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(v, str) and TOKEN_KEYS.search(k) and len(v) > 12:
                out[k] = v[:6] + "…" + v[-4:]
            else:
                out[k] = mask(v)
        return out
    if isinstance(obj, list):
        return [mask(v) for v in obj]
    if isinstance(obj, str):
        return re.sub(r"/mcp/job/v1\.[A-Za-z0-9._-]+", "/mcp/job/v1.<masked>", obj)
    return obj


class MCP:
    def __init__(self, url, log_path, client_name="worker", timeout=60):
        self.url = url
        self.log_path = log_path
        self.timeout = timeout
        self.session = None
        self.next_id = 1
        self.ctx = _ssl_context()
        self.client_name = client_name

    def log(self, kind, payload):
        with open(self.log_path, "a") as f:
            f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "kind": kind,
                                "data": mask(payload)}, ensure_ascii=False) + "\n")

    def _post(self, body):
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": "2025-06-18",
            "User-Agent": "mega-s02e04-" + self.client_name,
        }
        if self.session:
            headers["Mcp-Session-Id"] = self.session
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers=headers, method="POST")
        self.log("request", body)
        t0 = time.time()
        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout, context=self.ctx)
        except urllib.error.HTTPError as e:
            text = e.read().decode(errors="replace")
            self.log("http_error", {"status": e.code, "body": text[:4000]})
            raise RuntimeError(f"HTTP {e.code}: {text[:500]}")
        sid = resp.headers.get("Mcp-Session-Id")
        if sid:
            self.session = sid
        ctype = resp.headers.get("Content-Type", "")
        raw = resp.read().decode(errors="replace")
        dur = round(time.time() - t0, 2)
        if not raw.strip():
            self.log("reply", {"status": resp.status, "seconds": dur, "empty": True})
            return None
        if "text/event-stream" in ctype:
            msg = None
            for line in raw.splitlines():
                if line.startswith("data:"):
                    try:
                        cand = json.loads(line[5:].strip())
                    except ValueError:
                        continue
                    if isinstance(cand, dict) and cand.get("id") == body.get("id"):
                        msg = cand
            if msg is None:
                self.log("reply_unparsed", {"seconds": dur, "raw": raw[:4000]})
                raise RuntimeError("No JSON-RPC reply in event stream")
        else:
            msg = json.loads(raw)
        self.log("reply", {"status": resp.status, "seconds": dur, "body": msg})
        return msg

    def request(self, method, params=None):
        body = {"jsonrpc": "2.0", "id": self.next_id, "method": method}
        self.next_id += 1
        if params is not None:
            body["params"] = params
        msg = self._post(body)
        if msg is None:
            raise RuntimeError(f"Empty reply to {method}")
        if "error" in msg:
            raise RuntimeError(f"{method} error: {json.dumps(msg['error'])[:500]}")
        return msg.get("result")

    def notify(self, method, params=None):
        body = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            body["params"] = params
        self._post(body)

    def initialize(self):
        res = self.request("initialize", {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "mega-s02e04-" + self.client_name, "version": "1"},
        })
        self.notify("notifications/initialized")
        return res

    def call(self, name, arguments):
        """Call a tool; return (structured_or_parsed, raw_result)."""
        res = self.request("tools/call", {"name": name, "arguments": arguments})
        data = res.get("structuredContent")
        if data is None:
            texts = [c.get("text", "") for c in res.get("content", []) if c.get("type") == "text"]
            joined = "\n".join(texts)
            try:
                data = json.loads(joined)
            except ValueError:
                data = joined
        if res.get("isError"):
            raise RuntimeError(f"tool {name} returned an error: {json.dumps(data)[:800]}")
        return data, res
