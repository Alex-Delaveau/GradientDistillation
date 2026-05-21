#!/bin/bash
# Pré-télécharge DINOv2 ViT-B sur Jean Zay (à lancer depuis le frontal).
# Le frontal a accès à internet ; les nœuds de calcul non.

set -e

# Caches — doivent matcher ceux du script SLURM
export TORCH_HOME=${TORCH_HOME:-$WORK/torch_cache}
export HF_HOME=${HF_HOME:-$WORK/hf_cache}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$WORK/.cache}

mkdir -p "$TORCH_HOME" "$HF_HOME" "$XDG_CACHE_HOME"

echo "=== Cache configuration ==="
echo "TORCH_HOME      = $TORCH_HOME"
echo "HF_HOME         = $HF_HOME"
echo "XDG_CACHE_HOME  = $XDG_CACHE_HOME"
echo ""

# Vérifier qu'on est bien dans le venv
if [ -z "$VIRTUAL_ENV" ]; then
    echo "ERROR: no virtualenv activated."
    echo "Run: source \$WORK/GradientDistillation/.venv/bin/activate"
    exit 1
fi

echo "=== Downloading DINOv2 ViT-B/14 ==="
python <<'PYEOF'
import torch
import os

print(f"PyTorch version: {torch.__version__}")
print(f"TORCH_HOME: {os.environ.get('TORCH_HOME')}")

# Télécharge le repo (clone GitHub) + le checkpoint (.pth)
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', trust_repo=True)
print(f"Model loaded: {type(model).__name__}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
PYEOF

echo ""
echo "=== Verifying cache content ==="
echo "Repo clone:"
ls -la "$TORCH_HOME/hub/" 2>/dev/null | grep -E "dinov2|facebookresearch" || echo "  (not found)"
echo ""
echo "Checkpoint files:"
ls -lh "$TORCH_HOME/hub/checkpoints/" 2>/dev/null || echo "  (no checkpoints dir yet)"

echo ""
echo "✓ DINOv2 ViT-B/14 ready in $TORCH_HOME"
echo ""
echo "To add more models later, edit this script and re-run from the frontend."