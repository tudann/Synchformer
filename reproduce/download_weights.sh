#!/usr/bin/env bash
# Prefetch Synchformer sync checkpoints into ../logs/sync_models/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"
cd "$REPO"

# Prefer venv python if present.
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

python - <<'PY'
from pathlib import Path
from utils.utils import check_if_file_exists_else_download

exps = [
    "24-01-04T16-39-21",  # AudioSet — general AV sync (default for D16 pilot)
    "23-12-23T18-33-57",  # LRS3 — lip-oriented
]
for exp in exps:
    cfg = Path(f"./logs/sync_models/{exp}/cfg-{exp}.yaml")
    ckpt = Path(f"./logs/sync_models/{exp}/{exp}.pt")
    print(f"ensuring {cfg}")
    check_if_file_exists_else_download(str(cfg))
    print(f"ensuring {ckpt}")
    check_if_file_exists_else_download(str(ckpt))
    print(f"ok {exp}: cfg={cfg.stat().st_size}B ckpt={ckpt.stat().st_size}B")
print("done")
PY
