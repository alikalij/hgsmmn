# tests/integration/conftest.py
# ─── Step 1: mock ALL problematic imports FIRST ───────────────────────────────
import sys
from unittest.mock import MagicMock

_MOCKED = [
    "torch_cluster",
    "torch_sparse",
    "spconv",
    "spconv.pytorch",
    "flash_attn",
    "pointops",
    "addict",
    "wandb",
    "timm",
    "einops",
    "peft",
]
for _mod in _MOCKED:
    sys.modules.setdefault(_mod, MagicMock())

# ─── Step 2: now safe to touch sys.path ───────────────────────────────────────
import os

# مسیر پوشه‌ای که new_modules.py و point_transformer_v3m1_base.py در آن هستند
_HERE = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR = os.path.abspath(os.path.join(_HERE, "..", ".."))  # point_transformer_v3/

if _MODEL_DIR not in sys.path:
    sys.path.insert(0, _MODEL_DIR)
