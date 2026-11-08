# mega-s02e04 — two places

MEGA.dev Challenge S02E04. Two jobs, two workers in two places:

- **Job 1 (incident-lookup)** runs on the student's Mac.
- **Job 2 (voice-to-memo)** runs here, on GitHub Actions (`.github/workflows/job2.yml`, `worker2/`).

The main agent (the student's Claude harness) does neither job: it reads both results and joins them.

## How a job 2 run starts

Change the first word of `mode.txt` and push:

| mode | what it does |
|---|---|
| `probe` | read-only look: the job's tools and skill, a one-word test of the AI models, the clip's transcript |
| `run` | skill steps 1–3: transcribe, one note per finding, confirm; stops before submit |
| `submit` | skill step 4: hands in the confirmed notes |

Secrets (repo settings): `JOB2_URL` (the job's MCP link) and `GEMINI_API_KEY`.

The runner's output goes to the `results` branch, encrypted with `worker2/harness.pub.pem`; only the harness can read it, because Actions logs and branches of a public repo are public.
