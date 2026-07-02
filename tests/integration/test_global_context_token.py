import pytest
import torch
import torch.nn as nn

import mock_dependencies  # noqa: F401

from pointcept.models.point_transformer_v3.new_modules import GlobalContextToken
from pointcept.models.utils.structure import Point
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import PointTransformerV3


# =========================================================
# Helpers
# =========================================================

def _find_gct_modules(model: nn.Module):
    #return [m for m in model.modules() if isinstance(m, GlobalContextToken)]
    return [m for m in model.modules() if m.__class__.__name__ == "GlobalContextToken"]


def _make_point(
    num_points: int,
    in_channels: int,
    batch_sizes: tuple,
    seed: int = 0,
    device: str = "cpu",
):
    assert sum(batch_sizes) == num_points
    torch.manual_seed(seed)
    return Point(
        coord=torch.randn(num_points, 3, device=device),
        feat=torch.randn(num_points, in_channels, device=device),
        offset=torch.tensor(batch_sizes, dtype=torch.long, device=device).cumsum(0),
        grid_size=0.01
    )


def _extract_output_tensor(output):
    """
    Defensive output extraction for model variants that may return:
    - Tensor
    - Point-like object with .feat
    - dict with one of: seg_logits / logits / feat
    """
    if torch.is_tensor(output):
        return output
    if hasattr(output, "feat") and torch.is_tensor(output.feat):
        return output.feat
    if isinstance(output, dict):
        for key in ("seg_logits", "logits", "feat"):
            if key in output and torch.is_tensor(output[key]):
                return output[key]
    raise AssertionError(f"Unsupported output type: {type(output)}")


# =========================================================
# Fixtures
# =========================================================

@pytest.fixture(scope="session")
def cpu_device():
    return "cpu"


@pytest.fixture(scope="session")
def base_cfg():
    return dict(
        in_channels=3,
        order=("z", "z-trans"),
        stride=(2, ),
        enc_depths=(1, 1),
        enc_channels=(16, 32),
        enc_num_head=(2, 4),
        enc_patch_size=(16, 16),
        dec_depths=(1,),
        dec_channels=(32,),
        dec_num_head=(4,),
        dec_patch_size=(16,),
        mlp_ratio=4,
        drop_path=0.0,
        enable_rpe=False,
        enable_flash=False,
    )


# =========================================================
# Component tests
# =========================================================

class TestGlobalContextTokenComponent:
    @pytest.fixture
    def gct(self, cpu_device):
        torch.manual_seed(0)
        return GlobalContextToken(channels=8, num_anchors=4).to(cpu_device)

    def test_preserves_shape_and_returns_same_point(self, gct, cpu_device):
        point = _make_point(16, 8, (8, 8), seed=0, device=cpu_device)
        original_shape = point.feat.shape

        out = gct(point)

        assert out is point
        assert out.feat.shape == original_shape
        assert torch.isfinite(out.feat).all()

    def test_mutates_features(self, gct, cpu_device):
        point = _make_point(16, 8, (8, 8), seed=1, device=cpu_device)
        before = point.feat.clone()

        gct(point)

        assert not torch.allclose(point.feat, before)

    def test_handles_multi_batch(self, gct, cpu_device):
        point = _make_point(15, 8, (5, 4, 6), seed=2, device=cpu_device)

        out = gct(point)

        assert out.feat.shape == (15, 8)
        assert torch.isfinite(out.feat).all()

    def test_handles_single_point_batches(self, gct, cpu_device):
        point = _make_point(4, 8, (1, 1, 1, 1), seed=3, device=cpu_device)

        out = gct(point)

        assert out.feat.shape == (4, 8)
        assert torch.isfinite(out.feat).all()

    def test_is_deterministic_in_eval(self, cpu_device):
        torch.manual_seed(42)
        gct = GlobalContextToken(channels=8, num_anchors=4).to(cpu_device).eval()

        p1 = _make_point(12, 8, (12,), seed=10, device=cpu_device)
        p2 = Point(
            coord=p1.coord.clone(),
            feat=p1.feat.clone(),
            offset=p1.offset.clone(),
        )

        with torch.no_grad():
            out1 = gct(p1).feat.clone()
            out2 = gct(p2).feat.clone()

        assert torch.allclose(out1, out2, atol=1e-6)

    def test_gradient_flow(self, gct, cpu_device):
        point = _make_point(10, 8, (10,), seed=4, device=cpu_device)
        point.feat.requires_grad_(True)
        leaf_feat = point.feat

        loss = gct(point).feat.mean()
        loss.backward()

        assert leaf_feat.grad is not None
        assert torch.isfinite(leaf_feat.grad).all()

        for name, param in gct.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No grad for parameter: {name}"
                assert torch.isfinite(param.grad).all()


# =========================================================
# Integration tests
# =========================================================

class TestPointTransformerV3GCTIntegration:
    def test_gct_modules_present_when_enabled(self, base_cfg, cpu_device):
        model = PointTransformerV3(
            **base_cfg,
            enable_gct=True,
            gct_num_anchors=4,
        ).to(cpu_device)

        gcts = _find_gct_modules(model)

        assert len(gcts) >= 1
        assert getattr(model, "enable_gct", False) is True

    def test_no_gct_modules_when_disabled(self, base_cfg, cpu_device):
        model = PointTransformerV3(
            **base_cfg,
            enable_gct=False,
        ).to(cpu_device)

        assert len(_find_gct_modules(model)) == 0

    @pytest.mark.parametrize("enable_gct", [True, False])
    def test_forward_output_shape_and_finite(self, base_cfg, cpu_device, enable_gct):
        extra = {"gct_num_anchors": 4} if enable_gct else {}
        model = PointTransformerV3(
            **base_cfg,
            enable_gct=enable_gct,
            **extra,
        ).to(cpu_device).eval()

        point = _make_point(64, 3, (32, 32), seed=5, device=cpu_device)

        with torch.no_grad():
            out = _extract_output_tensor(model(point))

        assert out.shape[0] == 64
        assert torch.isfinite(out).all()

    def test_eval_is_deterministic(self, base_cfg, cpu_device):
        torch.manual_seed(123)
        model = PointTransformerV3(
            **base_cfg,
            enable_gct=True,
            gct_num_anchors=4,
        ).to(cpu_device).eval()

        kwargs = dict(
            num_points=48,
            in_channels=3,
            batch_sizes=(24, 24),
            seed=7,
            device=cpu_device,
        )

        with torch.no_grad():
            out1 = _extract_output_tensor(model(_make_point(**kwargs))).clone()
            out2 = _extract_output_tensor(model(_make_point(**kwargs))).clone()

        assert torch.allclose(out1, out2, atol=1e-6)

    def test_gct_actually_called_during_forward(self, base_cfg, cpu_device):
        model = PointTransformerV3(
            **base_cfg,
            enable_gct=True,
            gct_num_anchors=4,
        ).to(cpu_device).eval()

        gct_modules = _find_gct_modules(model)
        assert gct_modules, "No GCT module found in model."

        call_count = {"n": 0}

        def _hook(module, inp, out):
            call_count["n"] += 1

        handles = [m.register_forward_hook(_hook) for m in gct_modules]
        try:
            point = _make_point(64, 3, (32, 32), seed=9, device=cpu_device)
            with torch.no_grad():
                model(point)
        finally:
            for h in handles:
                h.remove()

        assert call_count["n"] >= len(gct_modules)

    def test_e2e_backward_with_gct(self, base_cfg, cpu_device):
        torch.manual_seed(42)
        model = PointTransformerV3(
            **base_cfg,
            enable_gct=True,
            gct_num_anchors=4,
        ).to(cpu_device).train()

        point = _make_point(32, 3, (16, 16), seed=11, device=cpu_device)
        point.feat.requires_grad_(True)

        out = _extract_output_tensor(model(point))
        loss = out.mean()
        loss.backward()

        assert point.feat.grad is not None
        assert torch.isfinite(point.feat.grad).all()

        gct_params_with_grad = [
            name
            for name, p in model.named_parameters()
            if "gct" in name.lower() and p.requires_grad and p.grad is not None
        ]
        assert gct_params_with_grad, "No GCT parameter received gradient."

    def test_gct_affects_model_output(self, base_cfg, cpu_device):
        torch.manual_seed(999)
        model_with = PointTransformerV3(
            **base_cfg,
            enable_gct=True,
            gct_num_anchors=4,
        ).to(cpu_device).eval()

        torch.manual_seed(999)
        model_without = PointTransformerV3(
            **base_cfg,
            enable_gct=False,
        ).to(cpu_device).eval()

        p1 = _make_point(48, 3, (20, 28), seed=12, device=cpu_device)
        p2 = Point(
            coord=p1.coord.clone(),
            feat=p1.feat.clone(),
            offset=p1.offset.clone(),
            grid_size = 0.01
        )

        with torch.no_grad():
            out_with = _extract_output_tensor(model_with(p1))
            out_without = _extract_output_tensor(model_without(p2))

        assert out_with.shape == out_without.shape
        assert not torch.allclose(out_with, out_without), \
            "Outputs with and without GCT are unexpectedly identical."
