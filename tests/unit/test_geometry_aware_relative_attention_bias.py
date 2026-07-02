# file: tests/unit/test_geometry_aware_relative_attention_bias.py
import pytest
import torch
import torch.nn as nn
import mock_dependencies

from pointcept.models.point_transformer_v3.new_modules import GeometryAwareRelativeAttentionBias


# =========================================================
# Helpers
# =========================================================

def make_rel_pos(batch=2, k=8, dtype=torch.float32):
    torch.manual_seed(42)
    return torch.randn(batch, k, k, 3, dtype=dtype)


@pytest.fixture
def rel_pos():
    return make_rel_pos(4, 8)


class TestGeometryAwareRelativeAttentionBias:

    def test_instantiation_defaults(self):
        grab = GeometryAwareRelativeAttentionBias (num_heads=8)

        assert grab.num_heads == 8
        assert grab.use_distance is True
        assert grab.per_head is True
        assert torch.isclose(grab.alpha.detach(), torch.tensor(0.01))

    def test_custom_parameters(self):
        grab = GeometryAwareRelativeAttentionBias (
            num_heads=4,
            hidden_dim=64,
            use_distance=False,
            per_head=False,
            init_scale=0.05,
        )

        assert grab.mlp[0].in_features == 3
        assert grab.mlp[-1].out_features == 1
        assert grab.alpha.item() == pytest.approx(0.05)

    def test_mlp_structure(self):
        grab = GeometryAwareRelativeAttentionBias (num_heads=4)

        layers = list(grab.mlp.children())

        assert isinstance(layers[0], nn.Linear)
        assert isinstance(layers[1], nn.GELU)
        assert isinstance(layers[2], nn.Linear)
        assert isinstance(layers[3], nn.GELU)
        assert isinstance(layers[4], nn.Dropout)
        assert isinstance(layers[5], nn.Linear)


# =========================================================
# Forward Shape
# =========================================================

@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("k", [4, 8, 16])
@pytest.mark.parametrize("heads", [2, 4, 8])
def test_forward_shape(batch, k, heads):
    grab = GeometryAwareRelativeAttentionBias (num_heads=heads)

    rel_pos = make_rel_pos(batch, k)

    out = grab(rel_pos)

    assert out.shape == (batch, heads, k, k)
    assert torch.isfinite(out).all()


def test_dtype_conversion():
    grab = GeometryAwareRelativeAttentionBias (num_heads=4)

    rel_pos = make_rel_pos(2, 5, dtype=torch.float64)

    out = grab(rel_pos)

    assert out.dtype == torch.float32


# =========================================================
# Internal Logic
# =========================================================

def test_use_distance_feature():
    rel_pos = torch.zeros(1, 2, 2, 3)
    rel_pos[0, 0, 1] = torch.tensor([3.0, 4.0, 0.0])

    grab = GeometryAwareRelativeAttentionBias (num_heads=2, use_distance=True)

    captured = []

    def hook(module, inp, out):
        captured.append(inp[0].detach())

    handle = grab.mlp.register_forward_hook(hook)

    _ = grab(rel_pos)

    handle.remove()

    mlp_input = captured[0]

    assert mlp_input.shape[-1] == 4


def test_per_head_false_identical(rel_pos):
    grab = GeometryAwareRelativeAttentionBias (num_heads=4, per_head=False)
    grab.eval()

    out = grab(rel_pos)

    for h in range(1, 4):
        assert torch.allclose(out[:, 0], out[:, h], atol=1e-6)


def test_alpha_scaling(rel_pos):
    grab = GeometryAwareRelativeAttentionBias (num_heads=4)
    grab.eval()

    out1 = grab(rel_pos)

    with torch.no_grad():
        grab.alpha *= 2

    out2 = grab(rel_pos)

    assert torch.allclose(out2, out1 * 2, atol=1e-5)


# =========================================================
# Determinism
# =========================================================

def test_eval_deterministic(rel_pos):
    grab = GeometryAwareRelativeAttentionBias (num_heads=4)
    grab.eval()

    out1 = grab(rel_pos)
    out2 = grab(rel_pos)

    assert torch.allclose(out1, out2)


def test_train_mode_runs(rel_pos):
    grab = GeometryAwareRelativeAttentionBias (num_heads=4)
    grab.train()

    out = grab(rel_pos)

    assert torch.isfinite(out).all()


# =========================================================
# Edge Cases
# =========================================================

@pytest.mark.parametrize("value", [0.0, -10.0, 1e5])
def test_extreme_values(value):
    grab = GeometryAwareRelativeAttentionBias (num_heads=4)

    rel_pos = torch.ones(2, 4, 4, 3) * value

    out = grab(rel_pos)

    assert torch.isfinite(out).all()


def test_single_token():
    grab = GeometryAwareRelativeAttentionBias (num_heads=4)

    rel_pos = make_rel_pos(2, 1)

    out = grab(rel_pos)

    assert out.shape == (2, 4, 1, 1)


# =========================================================
# Gradients
# =========================================================

def test_gradient_flow():
    grab = GeometryAwareRelativeAttentionBias (num_heads=4)

    rel_pos = make_rel_pos(2, 6).requires_grad_(True)

    out = grab(rel_pos)

    loss = out.mean()

    loss.backward()

    assert rel_pos.grad is not None
    assert grab.alpha.grad is not None

    for p in grab.mlp.parameters():
        assert p.grad is not None


def test_gradient_explosion_guard():
    grab = GeometryAwareRelativeAttentionBias (num_heads=4)

    rel_pos = make_rel_pos(2, 8).requires_grad_(True)

    loss = grab(rel_pos).sum()

    loss.backward()

    for p in grab.parameters():
        if p.grad is not None:
            assert p.grad.abs().max() < 1e6


# =========================================================
# Integration
# =========================================================

def test_attention_integration():

    B, H, K, D = 2, 4, 8, 16

    grab = GeometryAwareRelativeAttentionBias (num_heads=H)

    rel_pos = make_rel_pos(B, K)

    bias = grab(rel_pos)

    q = torch.randn(B, H, K, D)
    k = torch.randn(B, H, K, D)

    scores = torch.einsum("bhqd,bhkd->bhqk", q, k)

    scores = scores + bias

    attn = torch.softmax(scores, dim=-1)

    assert attn.shape == (B, H, K, K)
    assert torch.allclose(attn.sum(-1), torch.ones_like(attn.sum(-1)))

def test_grab_default_forward():
    """تست حالت پیش‌فرض (استفاده از فاصله و بایاس مجزا برای هر هد)"""
    N_prime, K, num_heads = 2, 8, 4
    model = GeometryAwareRelativeAttentionBias(num_heads=num_heads)
    
    # ورودی: مختصات نسبی 3 بعدی
    rel_pos = torch.randn(N_prime, K, K, 3)
    out = model(rel_pos)
    
    # بررسی ابعاد خروجی
    assert out.shape == (N_prime, num_heads, K, K), f"Expected shape {(N_prime, num_heads, K, K)}, got {out.shape}"
    
    # بررسی پارامتر یادگیرنده alpha
    assert model.alpha.requires_grad
    assert model.mlp[0].in_features == 4 # چون use_distance=True است، بعد ورودی باید 4 باشد

def test_grab_no_distance():
    """تست حالتی که از فاصله اقلیدسی استفاده نمی‌شود"""
    N_prime, K, num_heads = 2, 8, 4
    model = GeometryAwareRelativeAttentionBias(num_heads=num_heads, use_distance=False)
    
    rel_pos = torch.randn(N_prime, K, K, 3)
    out = model(rel_pos)
    
    assert out.shape == (N_prime, num_heads, K, K)
    assert model.mlp[0].in_features == 3 # بدون محاسبه فاصله، بعد ورودی باید همان 3 باشد

def test_grab_shared_per_head():
    """تست حالتی که بایاس برای همه هدها یکسان است (per_head=False)"""
    N_prime, K, num_heads = 2, 8, 4
    model = GeometryAwareRelativeAttentionBias(num_heads=num_heads, per_head=False)
    
    rel_pos = torch.randn(N_prime, K, K, 3)
    out = model(rel_pos)
    
    assert out.shape == (N_prime, num_heads, K, K)
    
    # وقتی per_head=False است، مقادیر هد اول باید دقیقاً با هد دوم برابر باشد
    assert torch.allclose(out[:, 0, :, :], out[:, 1, :, :]), "Values across heads should be identical when per_head=False"

def test_grab_gradient_flow():
    """بررسی جریان داشتن گرادیان برای بک‌پروپگیشن"""
    N_prime, K, num_heads = 2, 4, 2
    model = GeometryAwareRelativeAttentionBias(num_heads=num_heads)
    
    rel_pos = torch.randn(N_prime, K, K, 3)
    out = model(rel_pos)
    
    # ایجاد یک لاس مصنوعی و انجام backward
    loss = out.sum()
    loss.backward()
    
    # بررسی اینکه گرادیان‌ها برای پارامتر آلفا و لایه‌های شبکه محاسبه شده باشند
    assert model.alpha.grad is not None
    assert model.mlp[0].weight.grad is not None    
