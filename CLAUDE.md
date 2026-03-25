# CLAUDE.md — Project Instructions

## ⚠️ Secrets & Privacy Rules (READ FIRST)
- NEVER read, print, or include the contents of `.env`, `.env.*`, or any `*secrets*` files in responses
- NEVER suggest hardcoding API keys, tokens, or passwords anywhere in code
- If you need to reference a secret, use the variable name only (e.g. `OPENAI_API_KEY`) — never the value
- Secrets live in `.env` (local) or the OS keyring — see "Secrets Management" section below

## Git Workflow
- ALWAYS check `.gitignore` before staging any files
- Never commit files that match `.gitignore` patterns
- The following must NEVER be committed under any circumstances:
  - `.env`, `.env.local`, `.env.*.local`
  - Any file matching `*secret*`, `*credential*`, `*token*`
  - `.venv/`, `__pycache__/`, `*.pyc`, `*.pyo`
  - `*.key`, `*.pem`, `*.p12`
- Make separate, atomic commits per logical change
- Run pre-commit hooks before every commit (see below)
- Commit message format: `<type>: <short description>`
  - Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`

## Pre-Commit Hooks
Pre-commit is configured in `.pre-commit-config.yaml` in the project root.
ALWAYS run `pre-commit run --all-files` before committing.

### Custom Hooks (document yours here)
# - `[hook-name]` — [what it does, when it runs]


### Secret Detection Hook (add this if not already present)
Uses `detect-secrets` to block accidental secret commits:
```yaml
- repo: https://github.com/Yelp/detect-secrets
  rev: v1.4.0
  hooks:
    - id: detect-secrets
      args: ['--baseline', '.secrets.baseline']
```

## Secrets Management
Secrets are stored in `.env` and loaded via `python-dotenv`. Never use any other method.

<!-- ### Setup
```bash
pip install python-dotenv
cp .env.example .env   # .env.example has keys with no values — IS committed
                        # .env has real values — is NEVER committed
```

### In code
```python
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")  # Never hardcode
```

### .env.example (commit this as a template)
```
OPENAI_API_KEY=
DATABASE_URL=
SECRET_KEY=
``` -->

### .gitignore must contain
```
.env
.env.*
!.env.example
```

## Code Style
- Python 3.11+
- Formatter: `black` (line length 88)
- Linter: `ruff`
- Type hints required on all public functions
- Docstrings: Google style

## Other Persistent Behaviours
- Always run `pre-commit run --all-files` before staging
- Never modify `requirements.txt` directly — use `pip-compile`
- Ask before refactoring anything outside the current task scope
- Never run `git push --force` without explicit instruction