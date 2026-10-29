#!/usr/bin/env bash
# Create a pip venv for Synchformer inference (no conda on this host).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"
VENV="$ROOT/.venv"

if [[ ! -d "$VENV" ]]; then
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -U pip wheel

# Torch CUDA wheel. Override with TORCH_INDEX if needed.
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"
python -m pip install torch torchvision torchaudio --index-url "$TORCH_INDEX"

# Official tip: av==9.0.0 is most score-stable, but needs ffmpeg headers to build.
# On this host (Python 3.12, no libav-dev) use a binary wheel; scores may differ slightly.
python -m pip install "av==9.0.0" 2>/dev/null || python -m pip install "av>=12"
python -m pip install omegaconf timm transformers einops scipy tqdm

echo
echo "venv ready: $VENV"
echo "activate:  source $ROOT/.venv/bin/activate"
echo "repo root: $REPO"
echo "smoke:     cd $REPO && python example.py --exp_name 24-01-04T16-39-21 --vid_path <mp4> --offset_sec 0 --device cuda:0"
