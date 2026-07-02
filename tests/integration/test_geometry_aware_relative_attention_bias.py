# tests/test_geometry_aware_relative_attention_bias.py

import pytest
import torch
import torch.nn.functional as F

from pointcept.models.point_transformer_v3.new_modules import (
    GeometryAwareRelativeAttentionBias,
)

# Constants for testing
NUM_HEADS = 4
N_WINDOWS = 2
K = 8


@pytest.fixture
def module():
    torch.manual_seed(0)
    return GeometryAwareRelativeAttentionBias(
        num_heads=NUM_HEADS,
        hidden_dim=32,
        use_distance=True,
        per_head=True
    )


@pytest.fixture
def rel_pos():
    torch.manual_seed(1)
    # Shape: (N_windows, K, K, 3)
    return torch.randn(N_WINDOWS, K, K, 3)


def test_instantiation():
    m = GeometryAwareRelativeAttentionBias(num_heads=NUM_HEADS)
    assert m.num_heads == NUM_HEADS
    assert hasattr(m, "alpha")
    assert hasattr(m, "mlp")


def test_output_shape(module, rel_pos):
    out = module(rel_pos)
    # Shape should be (N_windows, num_heads, K, K)
    assert out.shape == (N_WINDOWS, NUM_HEADS, K, K)


@pytest.mark.parametrize(
    "n_win, k, heads",
    [
        (1, 4, 2),
        (2, 8, 4),
        (3, 16, 8),
    ],
)
def test_output_shape_parametric(n_win, k, heads):
    torch.manual_seed(0)
    m = GeometryAwareRelativeAttentionBias(num_heads=heads)
    rp = torch.randn(n_win, k, k, 3)
    out = m(rp)
    assert out.shape == (n_win, heads, k, k)


def test_output_is_finite(module, rel_pos):
    with torch.no_grad():
        out = module(rel_pos)
    assert torch.isfinite(out).all()


def test_zero_input_is_finite(module):
    rp = torch.zeros(N_WINDOWS, K, K, 3)
    with torch.no_grad():
        out = module(rp)
    assert torch.isfinite(out).all()


def test_large_input_is_finite(module):
    rp = torch.randn(N_WINDOWS, K, K, 3) * 1e3
    with torch.no_grad():
        out = module(rp)
    assert torch.isfinite(out).all()


def test_bias_is_addable_to_attention_logits(module, rel_pos):
    head_dim = 16
    # Mock Q and K for local window attention: (N_windows, num_heads, K, head_dim)
    q = torch.randn(N_WINDOWS, NUM_HEADS, K, head_dim)
    k_tensor = torch.randn(N_WINDOWS, NUM_HEADS, K, head_dim)

    # Logits: (N_windows, num_heads, K, K)
    logits = q @ k_tensor.transpose(-2, -1)
    bias = module(rel_pos)

    out = logits + bias
    assert out.shape == (N_WINDOWS, NUM_HEADS, K, K)
    assert torch.isfinite(out).all()


def test_softmax_pipeline_is_finite(module, rel_pos):
    head_dim = 16
    q = torch.randn(N_WINDOWS, NUM_HEADS, K, head_dim)
    k_tensor = torch.randn(N_WINDOWS, NUM_HEADS, K, head_dim)
    v = torch.randn(N_WINDOWS, NUM_HEADS, K, head_dim)

    logits = q @ k_tensor.transpose(-2, -1)
    bias = module(rel_pos)
    
    attn_weights = F.softmax(logits + bias, dim=-1)
    out = attn_weights @ v

    assert attn_weights.shape == (N_WINDOWS, NUM_HEADS, K, K)
    assert out.shape == (N_WINDOWS, NUM_HEADS, K, head_dim)
    assert torch.isfinite(attn_weights).all()
    assert torch.isfinite(out).all()


def test_gradient_flows_to_input(module):
    rp = torch.randn(N_WINDOWS, K, K, 3, requires_grad=True)
    out = module(rp)
    out.mean().backward()

    assert rp.grad is not None
    assert torch.isfinite(rp.grad).all()


def test_gradient_flows_to_parameters(module, rel_pos):
    out = module(rel_pos)
    out.mean().backward()

    found = False
    for name, p in module.named_parameters():
        if p.requires_grad:
            found = True
            assert p.grad is not None, f"missing grad for {name}"
            assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"
    assert found


def test_window_independence(module):
    # Ensure changing one window doesn't affect another
    rp1 = torch.randn(1, K, K, 3)
    rp2 = torch.randn(1, K, K, 3)
    rp_combined = torch.cat([rp1, rp2], dim=0)

    out_combined = module(rp_combined)
    out_single = module(rp1)

    assert torch.allclose(out_combined[0], out_single[0], atol=1e-6)


def test_bfloat16_cast_is_finite(module, rel_pos):
    out = module(rel_pos)
    out_bf16 = out.to(torch.bfloat16)
    assert torch.isfinite(out_bf16.float()).all()
