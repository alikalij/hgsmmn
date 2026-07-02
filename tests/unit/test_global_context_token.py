# file: tests/unit/test_global_context_token.py
import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace
import mock_dependencies

from pointcept.models.point_transformer_v3.new_modules import GlobalContextToken

# Fallback for torch_scatter if not available in the test environment
import pointcept.models.point_transformer_v3.new_modules as modules

def scatter_mean(src, index, dim=0, dim_size=None):
    """CPU-safe fallback for scatter_mean"""
    if dim_size is None:
        dim_size = int(index.max()) + 1 if index.numel() > 0 else 0
    
    out = torch.zeros(dim_size, src.size(1), device=src.device, dtype=src.dtype)
    count = torch.zeros(dim_size, 1, device=src.device, dtype=src.dtype)
    
    out.index_add_(0, index, src)
    count.index_add_(0, index, torch.ones(src.size(0), 1, device=src.device, dtype=src.dtype))
    
    return out / count.clamp_min(1)

def scatter_max(src, index, dim=0, dim_size=None):
    """CPU-safe fallback for scatter_max - returns (values, indices)"""
    if dim_size is None:
        dim_size = int(index.max()) + 1 if index.numel() > 0 else 0
    
    out = torch.full(
        (dim_size, src.size(1)),
        torch.finfo(src.dtype).min,
        device=src.device,
        dtype=src.dtype
    )
    argmax = torch.full(
        (dim_size, src.size(1)),
        -1,
        device=src.device,
        dtype=torch.long
    )
    
    for i in range(src.size(0)):
        b = int(index[i].item())
        mask = src[i] > out[b]
        out[b][mask] = src[i][mask]
        argmax[b][mask] = i
    
    return out, argmax

# Patch namespace ماژول (نه __globals__)
modules.scatter_mean = scatter_mean
modules.scatter_max = scatter_max


# --- Helper Functions ---
def get_trainable_parameter_grads(module: nn.Module):
    return [p.grad for p in module.parameters() if p.requires_grad and p.grad is not None]

def make_point(feat: torch.Tensor, offset: torch.Tensor):
    return SimpleNamespace(feat=feat, offset=offset)

def clone_point(point):
    return SimpleNamespace(feat=point.feat.clone(), offset=point.offset.clone())

def build_batch_index_from_offset(offset: torch.Tensor, N: int, device=None):
    batch_idx = torch.zeros(N, dtype=torch.long, device=device or offset.device)
    B = offset.shape[0]
    if B > 1:
        batch_idx[offset[:-1]] = 1
    batch_idx = torch.cumsum(batch_idx, dim=0)
    return batch_idx


# --- Fixtures ---
@pytest.fixture
def sample_channels():
    return 64

@pytest.fixture
def sample_multi_batch_point(sample_channels):
    torch.manual_seed(0)
    feat = torch.randn(12, sample_channels)
    offset = torch.tensor([5, 12], dtype=torch.long)
    return make_point(feat, offset)

@pytest.fixture
def single_batch_point(sample_channels):
    torch.manual_seed(1)
    feat = torch.randn(10, sample_channels)
    offset = torch.tensor([10], dtype=torch.long)
    return make_point(feat, offset)


# --- Tests ---
class TestGlobalContextToken:
    
    # ==================== Instantiation ====================
    @pytest.mark.sanity
    def test_instantiation_default(self):
        module = GlobalContextToken(channels=64)
        assert isinstance(module, nn.Module)
        assert module.channels == 64
        assert module.semantic_anchors.shape == (4, 64)
        assert module.pool_proj[0].in_features == 128
        assert module.gate[0].in_features == 64
        assert module.gate[0].out_features == 16

    @pytest.mark.parametrize(
        "channels, num_anchors",
        [(16, 2), (32, 4), (64, 8), (128, 16)],
    )
    def test_instantiation_various_configs(self, channels, num_anchors):
        module = GlobalContextToken(channels=channels, num_anchors=num_anchors)
        assert module.channels == channels
        assert module.semantic_anchors.shape == (num_anchors, channels)
        assert module.pool_proj[0].in_features == channels * 2
        assert module.gate[0].out_features == channels // 4

    # ==================== Forward Pass & Shapes ====================
    @pytest.mark.sanity
    @pytest.mark.parametrize("N, C, B_sizes", [
        (50, 32, [50]),
        (100, 64, [40, 60]),
        (200, 128, [50, 70, 80]),
    ])
    def test_forward_preserves_shape_dtype_device(self, N, C, B_sizes):
        torch.manual_seed(0)
        module = GlobalContextToken(channels=C)
        feat = torch.randn(N, C, dtype=torch.float32)
        offset = torch.tensor(B_sizes, dtype=torch.long).cumsum(0)
        point = make_point(feat, offset)
        
        output = module(point)
        
        assert output.feat.shape == feat.shape
        assert output.feat.dtype == feat.dtype
        assert output.feat.device == feat.device
        assert torch.equal(output.offset, offset)

    @pytest.mark.logic
    def test_forward_modifies_point_feat_inplace_but_not_offset(self, sample_multi_batch_point):
        module = GlobalContextToken(channels=64)
        offset_before = sample_multi_batch_point.offset.clone()
        feat_tensor_id_before = id(sample_multi_batch_point.feat)

        output = module(sample_multi_batch_point)

        assert id(output.feat) != feat_tensor_id_before
        assert torch.equal(output.offset, offset_before)

    # ==================== Logic & White-box ====================
    @pytest.mark.logic
    def test_batch_index_construction_matches_offset(self, sample_multi_batch_point):
        N = sample_multi_batch_point.feat.shape[0]
        batch_idx = build_batch_index_from_offset(sample_multi_batch_point.offset, N)
        expected = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=torch.long)
        assert torch.equal(batch_idx.cpu(), expected)

    @pytest.mark.logic
    def test_gate_values_are_bounded(self, sample_multi_batch_point, sample_channels):
        module = GlobalContextToken(channels=sample_channels)
        captured = {}

        def hook(_module, _inputs, output):
            captured["raw_gate"] = output.detach().clone()

        handle = module.gate.register_forward_hook(hook)
        try:
            _ = module(sample_multi_batch_point)
        finally:
            handle.remove()

        raw_gate = captured["raw_gate"]
        alpha = torch.sigmoid(raw_gate)
        assert alpha.shape == (sample_multi_batch_point.feat.shape[0], 1)
        assert torch.all(alpha >= 0.0) and torch.all(alpha <= 1.0)

    @pytest.mark.logic
    def test_residual_gated_update_matches_manual_computation(self, sample_multi_batch_point, sample_channels):
        torch.manual_seed(0)
        module = GlobalContextToken(channels=sample_channels)

        point = clone_point(sample_multi_batch_point)
        feat = point.feat.clone()
        offset = point.offset
        N = feat.shape[0]
        B = offset.shape[0]
        batch_idx = build_batch_index_from_offset(offset, N, device=feat.device)

        mean_pool = scatter_mean(feat, batch_idx, dim=0, dim_size=B)
        max_pool, _ = scatter_max(feat, batch_idx, dim=0, dim_size=B)
        pooled = torch.cat([mean_pool, max_pool], dim=-1)
        global_summary = module.pool_proj(pooled)

        attn_weights = torch.softmax(
            torch.matmul(global_summary, module.semantic_anchors.T) / (module.channels ** 0.5), dim=-1
        )
        semantic_context = torch.matmul(attn_weights, module.semantic_anchors)
        expanded_context = semantic_context[batch_idx]

        alpha = torch.sigmoid(module.gate(feat))
        expected = feat + alpha * module.out_proj(expanded_context)

        out = module(point)
        assert torch.allclose(out.feat, expected, atol=1e-6)

    @pytest.mark.logic
    def test_feature_permutation_equivariance_within_batch(self, sample_channels):
        torch.manual_seed(123)
        module = GlobalContextToken(channels=sample_channels)
        feat = torch.randn(10, sample_channels)
        offset = torch.tensor([4, 10], dtype=torch.long)

        point = make_point(feat.clone(), offset.clone())
        out_original = module(clone_point(point)).feat

        perm_batch0 = torch.tensor([2, 0, 3, 1], dtype=torch.long)
        perm_batch1 = torch.tensor([4, 2, 5, 1, 3, 0], dtype=torch.long) + 4
        perm = torch.cat([perm_batch0, perm_batch1], dim=0)

        point_perm = make_point(feat[perm].clone(), offset.clone())
        out_perm = module(point_perm).feat

        assert torch.allclose(out_original[perm], out_perm, atol=1e-6)

    # ==================== Edge Cases ====================
    @pytest.mark.edge_case
    def test_zero_features_input(self, sample_channels):
        module = GlobalContextToken(channels=sample_channels)
        offset = torch.tensor([5], dtype=torch.long)
        feat = torch.zeros(5, sample_channels)
        point = make_point(feat, offset)
        
        output = module(point)
        assert output.feat.shape == feat.shape
        assert torch.isfinite(output.feat).all()
        assert not torch.allclose(output.feat, torch.zeros_like(output.feat))

    @pytest.mark.edge_case
    def test_unequal_batch_sizes(self):
        torch.manual_seed(0)
        module = GlobalContextToken(channels=16)
        feat = torch.randn(205, 16)
        offset = torch.tensor([5, 205], dtype=torch.long)
        point = make_point(feat, offset)
        
        output = module(point)
        assert output.feat.shape == (205, 16)
        assert torch.isfinite(output.feat).all()

    # ==================== Gradients ====================
    @pytest.mark.gradients
    def test_gradient_magnitude_is_reasonable(self):
        torch.manual_seed(42)
        module = GlobalContextToken(channels=32, num_anchors=4)
        
        feat = torch.randn(200, 32, requires_grad=True)
        offset = torch.tensor([80, 200], dtype=torch.long)
        point = make_point(feat, offset)
        
        output = module(point)
        loss = output.feat.mean()
        loss.backward()
        
        # Check parameter gradients
        for name, param in module.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                assert 1e-6 < grad_norm < 1e3, f"Parameter '{name}' gradient norm {grad_norm:.6f} is out of bounds."
        
        # Check input gradients
        assert feat.grad is not None
        input_grad_norm = feat.grad.norm().item()
        assert 1e-6 < input_grad_norm < 1e3

    @pytest.mark.logic
    def test_repeatability_in_eval_mode(self, sample_multi_batch_point):
        module = GlobalContextToken(channels=64)
        module.eval()
        
        point1 = clone_point(sample_multi_batch_point)
        point2 = clone_point(sample_multi_batch_point)
        
        output1 = module(point1)
        output2 = module(point2)
        
        assert torch.allclose(output1.feat, output2.feat, atol=1e-7)
