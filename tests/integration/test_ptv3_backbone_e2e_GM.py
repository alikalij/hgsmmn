"""
tests/integration/test_ptv3_full_e2e_cpu.py
Integration Test for PointTransformerV3 (Backbone) and DefaultSegmentorV2 (Wrapper)
"""

import sys
import types
import pytest
import torch
import numpy as np

# ==========================================
# 1. Polyfill from GPT (Crucial for CPU Execution)
# ==========================================
def _install_torch_scatter_polyfill():
    """Install a small CPU polyfill before importing pointcept modules."""
    if "torch_scatter" in sys.modules:
        return
    torch_scatter = types.ModuleType("torch_scatter")
    def segment_csr(src, indptr, reduce="sum"):
        if indptr.ndim != 1:
            raise ValueError("indptr must be a 1D tensor")
        outputs = []
        for start, end in zip(indptr[:-1].tolist(), indptr[1:].tolist()):
            segment = src[start:end]
            if segment.numel() == 0:
                outputs.append(src.new_zeros(src.shape[1:]))
                continue
            if reduce in ("sum", "add"): outputs.append(segment.sum(dim=0))
            elif reduce == "mean": outputs.append(segment.mean(dim=0))
            elif reduce == "min": outputs.append(segment.min(dim=0).values)
            elif reduce == "max": outputs.append(segment.max(dim=0).values)
            else: raise NotImplementedError(f"Unsupported reduce: {reduce}")
        if not outputs:
            return src.new_empty((0, *src.shape[1:]))
        return torch.stack(outputs, dim=0)
    
    torch_scatter.segment_csr = segment_csr
    sys.modules["torch_scatter"] = torch_scatter

_install_torch_scatter_polyfill()

# Import mock dependencies if available in your project
try:
    import mock_dependencies  # noqa
except ImportError:
    pass

from pointcept.models.builder import build_model
from pointcept.models.utils.structure import Point

# ==========================================
# 2. Fixtures (From Claude)
# ==========================================
@pytest.fixture
def cpu_device():
    return torch.device('cpu')

@pytest.fixture
def full_config(cpu_device):
    """Production-like config with all new modules enabled, scaled for CPU."""
    return {
        "type": "DefaultSegmentorV2",
        "num_classes": 20,
        "backbone_out_channels": 64,
        "backbone": {
            "type": "PT-v3m1",
            "in_channels": 6,
            "order": ["z", "z-trans"],
            "stride": [2, 2],
            "enc_depths": [1, 1, 1],
            "enc_channels": [16, 32, 64],
            "enc_num_head": [2, 4, 8],
            "enc_patch_size": [32, 32, 32],
            "dec_depths": [1, 1],
            "dec_channels": [32, 16],
            "dec_num_head": [4, 2],
            "dec_patch_size": [32, 32],
            "mlp_ratio": 2,
            "qkv_bias": True,
            "drop_path": 0.0,
            "pre_norm": True,
            "shuffle_orders": False,
            "enable_rpe": False,  # CPU-safe
            "enable_flash": False,  # CPU-safe
            "upcast_attention": True,
            "upcast_softmax": True,
            # Feature Flags Activated
            "enable_spe": True,
            "spe_dim": 16,
            "enable_gsc": True,
            "enable_grab": True,
            "grab_hidden_dim": 16,
            "grab_use_distance": True,
            "grab_per_head": True,
            "grab_init_scale": 0.01,
            "enable_gtp": True,
            "gtp_prune_ratio": [0.1, 0.2],
            "gtp_k": 4,
            "enable_gct": True,
            "gct_num_anchors": 2,
            "pdnorm_decouple": True,
            "pdnorm_affine": True,
            "pdnorm_conditions": ["ScanNet"],
        },
        "criteria": [
            {"type": "CrossEntropyLoss", "loss_weight": 1.0, "ignore_index": -1},
        ],
    }

@pytest.fixture
def dummy_point_cloud(cpu_device):
    batch_size, num_points_per_batch = 2, 256
    total_points = batch_size * num_points_per_batch
    return {
        "coord": torch.randn(total_points, 3, device=cpu_device) * 10.0,
        "feat": torch.randn(total_points, 6, device=cpu_device),
        "offset": torch.tensor([num_points_per_batch * (i + 1) for i in range(batch_size)], dtype=torch.long, device=cpu_device),
        "grid_coord": (torch.randn(total_points, 3, device=cpu_device) * 50).long(),
        "segment": torch.randint(0, 20, (total_points,), dtype=torch.long, device=cpu_device),
        "condition": "ScanNet",
    }

# ==========================================
# 3. Test Suite (Combined & Improved)
# ==========================================
class TestPointTransformerV3Integration:
    
    def test_direct_backbone_contract(self, full_config, dummy_point_cloud, cpu_device):
        """تست مستقیم Backbone (رویکرد ترکیبی برای اطمینان از عملکرد خود کلاس)"""
        backbone_cfg = full_config["backbone"].copy()
        backbone = build_model(backbone_cfg).to(cpu_device).eval()
        
        with torch.no_grad():
            output = backbone(dummy_point_cloud)
        
        # بررسی خروجی مستقیم Backbone
        assert isinstance(output, Point), "Backbone باید شیء Point برگرداند."
        assert hasattr(output, "feat"), "شیء Point باید دارای feat باشد."
        assert torch.isfinite(output.feat).all(), "خروجی Backbone حاوی NaN/Inf است."
        assert output.feat.shape[1] == full_config["backbone_out_channels"]

    def test_forward_train_and_loss(self, full_config, dummy_point_cloud, cpu_device):
        """تست E2E روی Segmentor: محاسبه Loss در حالت Train"""
        model = build_model(full_config).to(cpu_device).train()
        output = model(dummy_point_cloud)
        
        assert isinstance(output, dict)
        assert "loss" in output
        assert torch.isfinite(output["loss"]), "مقدار Loss نامعتبر است."

    def test_backward_and_finite_gradients(self, full_config, dummy_point_cloud, cpu_device):
        """تست Backward با بررسی سخت‌گیرانه گرادیان‌ها (از GPT)"""
        model = build_model(full_config).to(cpu_device).train()
        output = model(dummy_point_cloud)
        output["loss"].backward()
        
        finite_grads, nonzero_grads = 0, 0
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert torch.isfinite(param.grad).all(), f"Gradient انفجاری در {name}"
                finite_grads += 1
                if param.grad.abs().sum().item() > 0:
                    nonzero_grads += 1
                    
        assert finite_grads > 0, "هیچ گرادیانی محاسبه نشد!"
        assert nonzero_grads > 0, "تمام گرادیان‌ها صفر هستند!"

    def test_new_modules_activation_hooks(self, full_config, dummy_point_cloud, cpu_device):
        """بررسی فعال شدن واقعی ماژول‌های جدید در جریان Forward (از Claude)"""
        model = build_model(full_config).to(cpu_device).eval()
        activations = set()
        
        def hook_fn(name):
            def hook(module, input, output): activations.add(name)
            return hook
            
        for name, module in model.named_modules():
            if any(k in module.__class__.__name__.lower() for k in ['spe', 'gct', 'gtp', 'grab', 'gsc']):
                module.register_forward_hook(hook_fn(name))
                
        with torch.no_grad():
            model(dummy_point_cloud)
            
        assert len(activations) > 0, "ماژول‌های جدید (SPE/GCT/...) فعال نشدند!"

    def test_deterministic_output(self, full_config, dummy_point_cloud, cpu_device):
        """تست پایداری و Deterministic بودن خروجی"""
        torch.manual_seed(42)
        model = build_model(full_config).to(cpu_device).eval()
        
        with torch.no_grad():
            torch.manual_seed(123)
            out1 = model(dummy_point_cloud)["seg_logits"]
            torch.manual_seed(123)
            out2 = model(dummy_point_cloud)["seg_logits"]
            
        torch.testing.assert_close(out1, out2, rtol=1e-5, atol=1e-5)
