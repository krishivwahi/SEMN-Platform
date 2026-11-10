#!/usr/bin/env bash
# Installs ponytail, rederive, pstack (cursor/plugins) and a garak skill at user scope, for every project.
# Idempotent; also the cloud setup script. Usage: bash tools/claude/install.sh [--garak to install garak now]
set -u; export PATH="$PATH:/opt/claude-code/bin"  # where cloud containers keep the claude CLI
for m in DietrichGebert/ponytail EricJujianZou/rederive cursor/plugins; do
  claude plugin marketplace add "$m" || echo "warn: could not add marketplace $m"
done
for p in ponytail@ponytail rederive@rederive pstack@cursor-plugins; do
  claude plugin install "$p" || echo "warn: could not install $p"
done
mkdir -p "$HOME/.claude/skills/garak"
cat > "$HOME/.claude/skills/garak/SKILL.md" <<'SKILL'
---
name: garak
description: Red-team an LLM, chatbot, agent or LLM API with NVIDIA garak (prompt injection, jailbreaks, data leaks). Use when asked to security-test a model, or when garak is mentioned.
---

# garak: LLM vulnerability scanning

garak sends attack prompts (probes) to a model, scores replies with detectors and writes a report.
It tests a model or endpoint, not source code: if the project has no LLM in it, say so instead of scanning.

## 0. Install on first use

Run `garak --version`. If it is missing, install it in its own venv (several GB, clashes with project deps):

```bash
python3 -m venv ~/.local/share/garak-venv
~/.local/share/garak-venv/bin/pip install -U pip setuptools wheel
~/.local/share/garak-venv/bin/pip install garak
mkdir -p ~/.local/bin && ln -sf ~/.local/share/garak-venv/bin/garak ~/.local/bin/garak
```

On Windows use `py -m venv`, `Scripts\pip` and `Scripts\garak` instead. Never add garak to project requirements unasked.

## 1. Confirm scope first

- Only scan models and endpoints the user owns or is authorized to test. Ask before touching anything third-party.
- A scan can send thousands of requests: real money on a paid API, and rate limits.
  State the rough count (prompts x `--generations`) and get a yes before a large run.
- Keys come from env vars: `OPENAI_API_KEY`, `OPENAICOMPATIBLE_API_KEY`, `ANTHROPIC_API_KEY` (via LiteLLM),
  `HF_INFERENCE_TOKEN`, `REST_API_KEY`. Never write a key into a repo file or the chat.

## 2. Target: `-t <type> -n <name>` (`garak --list_generators` lists all)

- OpenAI: `-t openai -n gpt-4o-mini`
- Anthropic and most providers via LiteLLM: `-t litellm -n anthropic/<model-id>`
- OpenAI-compatible server (vLLM, LM Studio, a gateway): `-t openai.OpenAICompatible -n <model>`
  plus `--generator_options '{"openai":{"OpenAICompatible":{"uri":"http://localhost:8000/v1"}}}'`
- Local: `-t ollama -n llama3.2`, or `-t huggingface -n gpt2`
- Dry run with no model: `-t test.Blank -S probes.test.Test`
- The user's own app over HTTP: `-t rest -G rest.json`, with rest.json in a scratch dir, not the repo:

```json
{"rest": {"RestGenerator": {
  "uri": "http://localhost:8000/chat", "method": "post",
  "headers": {"Authorization": "Bearer $KEY", "Content-Type": "application/json"},
  "req_template_json_object": {"message": "$INPUT"},
  "response_json": true, "response_json_field": "reply"
}}}
```
`$INPUT` becomes each attack prompt; `$KEY` becomes the `REST_API_KEY` env var.

## 3. Probes: `-S <spec>` (`--probes` is deprecated)

Spec syntax: `probes.<module>[.<Class>]`, `tag:<prefix>`, `tier:<N>`; prefix `-` to exclude; comma-separated.
`garak --list_probes -v` describes every probe. Start narrow with `-g 1`; widen once a first run works.

- Prompt injection: `probes.promptinject,probes.latentinjection` (or `tag:owasp:llm01`)
- Jailbreaks: `probes.dan,-probes.dan.DanInTheWild`
- System prompt leakage: `probes.sysprompt_extraction`
- Data and secret leakage: `probes.leakreplay,probes.apikey`
- Unsafe code: `probes.malwaregen,probes.packagehallucination,probes.exploitation`
- Encoding tricks: `probes.encoding,probes.smuggling,probes.ansiescape`
- Broad but bounded: `tier:1`

## 4. Run

```bash
garak -t <type> -n <name> -S <spec> -g 1 --report_prefix <short-name> --parallel_attempts 4
```

Run anything beyond a smoke test in the background. Output goes to `~/.local/share/garak/garak_runs/`:
`<prefix>.report.jsonl`, `.report.html` and `.hitlog.jsonl` (one line per failing reply).

## 5. Report back

1. Table of probe, detector and pass rate, worst first (from lines like `probe  detector: FAIL ok on X/Y`).
2. Two or three failing prompt/reply pairs from the hitlog, trimmed.
3. The OWASP LLM Top 10 item each failure maps to, and a fix: input filtering, system prompt change,
   output validation, tighter tool permissions, or a different model.
4. The path to the HTML report.
5. Detectors are heuristics: spot-check hits before calling one a vulnerability, and flag likely false positives.
SKILL
if [ "${1:-}" = "--garak" ] && ! command -v garak >/dev/null; then
  python3 -m venv "$HOME/.local/share/garak-venv"
  "$HOME/.local/share/garak-venv/bin/pip" install -q -U pip setuptools wheel
  "$HOME/.local/share/garak-venv/bin/pip" install -q garak
  mkdir -p "$HOME/.local/bin" && ln -sf "$HOME/.local/share/garak-venv/bin/garak" "$HOME/.local/bin/garak"
fi
