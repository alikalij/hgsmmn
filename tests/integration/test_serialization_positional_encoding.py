# tests/integration/test_spe_integration.py

import pytest
import torch
import torch.nn as nn
from copy import deepcopy

from pointcept.models.point_transformer_v3.new_modules import (
    SerializationPositionalEncoding,
)
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import (
    PointTransformerV3,
    SerializedAttention,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

class _SpySPE(nn.Module):
    """
    Wraps a real SPE and counts forward calls.
    Preferred over monkeypatch on bound methods (PyTorch C++ dispatch unsafe).
    """
    def __init__(self, real: SerializationPositionalEncoding):
        super().__init__()
        self.real = real
        self.call_count = 0

    def forward(self, features: torch.Tensor, serialized_order: torch.Tensor):
        assert features.ndim == 2,          "SPE expects 2-D feature tensor"
        assert serialized_order.ndim == 1,  "SPE expects 1-D order tensor"
        assert features.shape[0] == serialized_order.shape[0]
        self.call_count += 1
        return self.real(features, serialized_order)


class _ZeroSPE(nn.Module):
    """Returns features unchanged — used to verify SPE actually affects output."""
    def forward(self, features, serialized_order):
        return features


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def cpu_device():
    return torch.device("cpu")


@pytest.fixture
def model_config():
    """Minimal config — cpu-friendly, no flash, no RPE."""
    return dict(
        in_channels=3,
        order=("z", "z-trans"),
        stride=(2, 2),
        enc_depths=(1, 1, 1),
        enc_channels=(8, 16, 32),
        enc_num_head=(2, 4, 8),
        enc_patch_size=(8, 8, 8),
        dec_depths=(1, 1),
        dec_channels=(16, 8),
        dec_num_head=(4, 2),
        dec_patch_size=(8, 8),
        drop_path=0.0,
        shuffle_orders=False,
        enable_flash=False,
        enable_rpe=False,
        enable_spe=True,
        spe_dim=16,
    )


@pytest.fixture
def dummy_input(cpu_device):
    """
    grid_size is required by point.serialization() in structure.py:
        assert {"grid_size", "coord"}.issubset(self.keys())
    Omitting it causes AssertionError before any model logic runs.
    """
    torch.manual_seed(42)
    N = 1024
    return {
        "coord":      torch.randn(N, 3, device=cpu_device),
        "feat":       torch.randn(N, 3, device=cpu_device),
        "grid_coord": torch.zeros(N, 3, dtype=torch.long, device=cpu_device),
        "offset":     torch.tensor([N], dtype=torch.long, device=cpu_device),
        "grid_size":  0.01,   # ← required; absence → AssertionError in serialization
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Construction tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestPTV3Construction:

    def test_spe_module_exists_when_enabled(self, cpu_device, model_config):
        model = PointTransformerV3(**model_config).to(cpu_device)
        assert hasattr(model, "spe")
        assert isinstance(model.spe, SerializationPositionalEncoding)

    def test_spe_dims_match_config(self, cpu_device, model_config):
        model = PointTransformerV3(**model_config).to(cpu_device)
        assert model.spe.channels == model_config["enc_channels"][0]
        assert model.spe.hidden_dim == model_config["spe_dim"]

    def test_spe_absent_when_disabled(self, cpu_device, model_config):
        cfg = {**model_config, "enable_spe": False}
        model = PointTransformerV3(**cfg).to(cpu_device)
        # getattr guard: attribute may not exist at all when disabled
        assert getattr(model, "spe", None) is None
        assert getattr(model, "spe_modules", None) is None

    def test_stagewise_spe_modules_wiring(self, cpu_device, model_config):
        """If spe_modules is built, each entry must match the corresponding stage dims."""
        model = PointTransformerV3(**model_config).to(cpu_device)
        if not hasattr(model, "spe_modules"):
            pytest.skip("spe_modules not present in this version — structural contract pending")

        assert isinstance(model.spe_modules, nn.ModuleList)
        assert len(model.spe_modules) == len(model_config["enc_depths"])
        for i, mod in enumerate(model.spe_modules):
            assert isinstance(mod, SerializationPositionalEncoding)
            assert mod.channels == model_config["enc_channels"][i]
            assert mod.hidden_dim == model_config["spe_dim"]

    def test_stage_wrapper_not_inserted_yet(self, cpu_device, model_config):
        """Contract: SPEStageWrapper is commented out — documents current absence."""
        model = PointTransformerV3(**model_config).to(cpu_device)
        found = any(type(m).__name__ == "SPEStageWrapper" for m in model.modules())
        assert found is False, "SPEStageWrapper appeared — update or remove this contract test"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Active forward-path tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestPTV3ActiveForward:

    def test_spe_called_exactly_once_per_forward(self, cpu_device, model_config, dummy_input):
        model = PointTransformerV3(**model_config).eval().to(cpu_device)
        # Direct attribute replacement — safe, no monkeypatch on bound method
        model.spe = _SpySPE(model.spe)

        with torch.no_grad():
            model(deepcopy(dummy_input))

        assert model.spe.call_count == 1

    def test_output_is_finite(self, cpu_device, model_config, dummy_input):
        model = PointTransformerV3(**model_config).eval().to(cpu_device)
        with torch.no_grad():
            out = model(deepcopy(dummy_input))
        assert out.feat.ndim == 2
        assert out.feat.shape[0] > 0
        assert torch.isfinite(out.feat).all()

    def test_eval_is_deterministic(self, cpu_device, model_config, dummy_input):
        model = PointTransformerV3(**model_config).eval().to(cpu_device)
        inp = deepcopy(dummy_input)
        with torch.no_grad():
            o1 = model(deepcopy(inp)).feat.clone()
            o2 = model(deepcopy(inp)).feat.clone()
        assert torch.allclose(o1, o2, atol=1e-6)

    def test_spe_ablation_changes_output(self, cpu_device, model_config, dummy_input):
        """
        Dead-code guard: if SPE were bypassed, output must differ.
        Failure means SPE is wired but not actually consumed (dead code).
        """
        model = PointTransformerV3(**model_config).eval().to(cpu_device)
        inp = deepcopy(dummy_input)

        with torch.no_grad():
            feat_real = model(deepcopy(inp)).feat.clone()
            model.spe = _ZeroSPE()
            feat_zero = model(deepcopy(inp)).feat.clone()

        assert not torch.allclose(feat_real, feat_zero, atol=1e-6), (
            "SPE is dead — output identical with ZeroSPE. "
            "Check that forward() actually consumes model.spe."
        )

    def test_gradients_reach_spe_parameters(self, cpu_device, model_config, dummy_input):
        model = PointTransformerV3(**model_config).eval().to(cpu_device)
        inp = deepcopy(dummy_input)
        inp["feat"] = inp["feat"].requires_grad_(True)

        model(inp).feat.mean().backward()

        spe_grads = [
            p.grad for p in model.spe.parameters()
            if p.requires_grad and p.grad is not None
        ]
        assert len(spe_grads) > 0, "No gradients reached SPE parameters"
        assert all(torch.isfinite(g).all() for g in spe_grads), "NaN/Inf in SPE gradients"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SerializedAttention — active (disabled SPE) + xfail (enabled SPE)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSerializedAttentionSPE:

    def test_disabled_spe_is_none(self, cpu_device):
        attn = SerializedAttention(
            channels=8, num_heads=2, patch_size=8,
            qkv_bias=True, attn_drop=0.0, proj_drop=0.0,
            order_index=0, enable_rpe=False, enable_flash=False,
            enable_spe=False,
        ).to(cpu_device)
        assert getattr(attn, "spe", None) is None

    @pytest.mark.xfail(
        raises=TypeError,
        strict=True,
        reason=(
            "SerializedAttention passes wrong kwargs to SPE constructor. "
            "SPE expects (channels, hidden_dim) but receives (in_channels, out_channels, ...). "
            "Fix: align SerializedAttention.__init__ with SPE API."
        ),
    )
    def test_enabled_spe_constructor_mismatch(self, cpu_device):
        SerializedAttention(
            channels=8, num_heads=2, patch_size=8,
            qkv_bias=True, attn_drop=0.0, proj_drop=0.0,
            order_index=0, enable_rpe=False, enable_flash=False,
            enable_spe=True, spe_dim=8,
        ).to(cpu_device)

    @pytest.mark.xfail(
        strict=False,
        reason="SPE injection inside SerializedAttention.forward is commented out in production.",
    )
    def test_forward_spe_path_inactive(self, cpu_device):
        """Documents that the SPE forward path inside attention is not yet active."""
        attn = SerializedAttention(
            channels=8, num_heads=2, patch_size=8,
            qkv_bias=True, attn_drop=0.0, proj_drop=0.0,
            order_index=0, enable_rpe=False, enable_flash=False,
            enable_spe=False,
        ).to(cpu_device)
        # Manually inject SPE and a spy to detect if forward uses it
        attn.spe = _SpySPE(
            SerializationPositionalEncoding(channels=8, hidden_dim=8).to(cpu_device)
        )
        attn.enable_spe = True

        N = 16
        class _FakePoint:
            feat = torch.randn(N, 8, device=cpu_device)
            offset = torch.tensor([N], dtype=torch.long, device=cpu_device)
            grid_coord = torch.zeros(N, 3, dtype=torch.long, device=cpu_device)
            serialized_order = [torch.arange(N, device=cpu_device)]
            serialized_inverse = [torch.arange(N, device=cpu_device)]

        with torch.no_grad():
            attn.eval()
            attn(_FakePoint())

        # This assert will fail until forward path is un-commented
        assert attn.spe.call_count == 1


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Commented-path tests — activated via patching inside the test
#    (from Gemini's patched_forward idea, cleaned up)
#    Remove/merge these once production code is un-commented.
# ═══════════════════════════════════════════════════════════════════════════════

class TestCommentedPathsActivated:
    """
    Simulates commented-out code paths by patching inside the test scope only.
    No production files are modified.
    When production code is un-commented, delete these and use the direct tests above.
    """

    class _PatchedSerializedAttention(SerializedAttention):
        """
        SerializedAttention with two fixes applied:
          1. Constructor uses correct SPE API (channels, hidden_dim)
          2. forward() calls spe — mirrors the commented-out production path
        """
        def __init__(self, *args, enable_spe=False, spe_dim=16, **kwargs):
            super().__init__(*args, enable_spe=False, **kwargs)
            if enable_spe:
                self.spe = SerializationPositionalEncoding(
                    channels=kwargs.get("channels", args[0] if args else 8),
                    hidden_dim=spe_dim,
                )
                self.enable_spe = True
            else:
                self.spe = None
                self.enable_spe = False

        def forward(self, point):
            if self.enable_spe and self.spe is not None:
                # Mirrors the commented line in SerializedAttention.forward
                point.feat = self.spe(
                    point.feat,
                    point.serialized_order[self.order_index],
                )
            return super().forward(point)

    def test_patched_attn_constructs_spe_correctly(self, cpu_device):
        attn = self._PatchedSerializedAttention(
            channels=8, num_heads=2, patch_size=8,
            qkv_bias=True, attn_drop=0.0, proj_drop=0.0,
            order_index=0, enable_rpe=False, enable_flash=False,
            enable_spe=True, spe_dim=8,
        ).to(cpu_device)

        assert attn.enable_spe is True
        assert isinstance(attn.spe, SerializationPositionalEncoding)
        assert attn.spe.channels == 8
        assert attn.spe.hidden_dim == 8

    def test_patched_attn_forward_calls_spe(self, cpu_device):
        attn = self._PatchedSerializedAttention(
            channels=8, num_heads=2, patch_size=8,
            qkv_bias=True, attn_drop=0.0, proj_drop=0.0,
            order_index=0, enable_rpe=False, enable_flash=False,
            enable_spe=True, spe_dim=8,
        ).to(cpu_device)

        spy = _SpySPE(attn.spe)
        attn.spe = spy

        N = 16
        class _FakePoint:
            def __init__(self, f=None):
                N = 16 # در این تستِ ایزوله، N کوچک مشکلی ندارد
                self.feat = f.clone() if f is not None else torch.randn(N, 8, device=cpu_device)
                self.offset = torch.tensor([N], dtype=torch.long, device=cpu_device)
                self.grid_coord = torch.zeros(N, 3, dtype=torch.long, device=cpu_device)
                self.serialized_order = [torch.arange(N, device=cpu_device)]
                self.serialized_inverse = [torch.arange(N, device=cpu_device)]

            # شبیه‌سازی رفتار دیکشنری‌مانند کلاس Point اصلی
            def keys(self):
                return self.__dict__.keys()

            def __getitem__(self, key):
                return getattr(self, key)

            def __setitem__(self, key, value):
                setattr(self, key, value)

        attn.eval()
        with torch.no_grad():
            out = attn(_FakePoint())

        assert spy.call_count == 1
        assert torch.isfinite(out.feat).all()

    def test_patched_attn_output_differs_from_no_spe(self, cpu_device):
        """Ablation inside the patched path."""
        base_attn = self._PatchedSerializedAttention(
            channels=8, num_heads=2, patch_size=8,
            qkv_bias=True, attn_drop=0.0, proj_drop=0.0,
            order_index=0, enable_rpe=False, enable_flash=False,
            enable_spe=False,
        ).eval().to(cpu_device)

        spe_attn = self._PatchedSerializedAttention(
            channels=8, num_heads=2, patch_size=8,
            qkv_bias=True, attn_drop=0.0, proj_drop=0.0,
            order_index=0, enable_rpe=False, enable_flash=False,
            enable_spe=True, spe_dim=8,
        ).eval().to(cpu_device)

        N = 16
        feat = torch.randn(N, 8, device=cpu_device)
        order = torch.arange(N, device=cpu_device)

        class _FakePoint:
            def __init__(self, f=None):
                N = 16 # در این تستِ ایزوله، N کوچک مشکلی ندارد
                self.feat = f.clone() if f is not None else torch.randn(N, 8, device=cpu_device)
                self.offset = torch.tensor([N], dtype=torch.long, device=cpu_device)
                self.grid_coord = torch.zeros(N, 3, dtype=torch.long, device=cpu_device)
                self.serialized_order = [torch.arange(N, device=cpu_device)]
                self.serialized_inverse = [torch.arange(N, device=cpu_device)]

            # شبیه‌سازی رفتار دیکشنری‌مانند کلاس Point اصلی
            def keys(self):
                return self.__dict__.keys()
            
            def __getitem__(self, key):
                return getattr(self, key)

            def __setitem__(self, key, value):
                setattr(self, key, value)

        with torch.no_grad():
            out_no_spe = base_attn(_FakePoint(feat)).feat.clone()
            out_spe    = spe_attn(_FakePoint(feat)).feat.clone()

        assert not torch.allclose(out_no_spe, out_spe, atol=1e-6), (
            "SPE path has no effect — check _PatchedSerializedAttention.forward"
        )
