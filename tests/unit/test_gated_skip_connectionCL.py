# file: tests/unit/test_gated_skip_connectionCL.py
import io
import pytest
import torch
import torch.nn as nn
import mock_dependencies

from pointcept.models.point_transformer_v3.new_modules import GatedSkipConnectionCL


class TestGatedSkipConnectionCL:
    """Independent test suite for GatedSkipConnectionCL."""

    def make_inputs(self, n=16, enc_c=64, dec_c=64, seed=123):
        torch.manual_seed(seed)
        enc = torch.randn(n, enc_c)
        dec = torch.randn(n, dec_c)
        return enc, dec

    # =====================================================
    # Initialization
    # =====================================================

    def test_init_with_same_channels_uses_identity_projection(self):
        module = GatedSkipConnectionCL(enc_channels=64, dec_channels=64)

        assert isinstance(module.gate, nn.Sequential)
        assert isinstance(module.gate[0], nn.Linear)
        assert isinstance(module.gate[1], nn.LayerNorm)
        assert isinstance(module.gate[2], nn.ReLU)
        assert isinstance(module.gate[3], nn.Linear)
        assert isinstance(module.gate[4], nn.Sigmoid)

        assert isinstance(module.dec_proj, nn.Identity)

    def test_init_with_different_channels_uses_linear_projection(self):
        module = GatedSkipConnectionCL(enc_channels=128, dec_channels=64)

        assert isinstance(module.dec_proj, nn.Linear)
        assert module.dec_proj.in_features == 64
        assert module.dec_proj.out_features == 128

    def test_default_hidden_dim_uses_max_half_or_16_large_case(self):
        module = GatedSkipConnectionCL(enc_channels=128, dec_channels=64)
        assert module.gate[0].out_features == 64  # 128 // 2

    def test_default_hidden_dim_uses_minimum_16_small_case(self):
        module = GatedSkipConnectionCL(enc_channels=8, dec_channels=8)
        assert module.gate[0].out_features == 16

    def test_custom_hidden_dim_is_respected(self):
        module = GatedSkipConnectionCL(enc_channels=64, dec_channels=32, hidden_dim=40)
        assert module.gate[0].out_features == 40
        assert module.gate[1].normalized_shape == (40,)
        assert module.gate[3].in_features == 40
        assert module.gate[3].out_features == 64

    # =====================================================
    # Forward shape / interface
    # =====================================================

    @pytest.mark.parametrize(
        "n, enc_c, dec_c",
        [
            (1, 8, 8),
            (4, 16, 16),
            (8, 64, 32),
            (32, 128, 64),
        ],
    )
    def test_forward_output_shape_is_n_by_enc_channels(self, n, enc_c, dec_c):
        module = GatedSkipConnectionCL(enc_channels=enc_c, dec_channels=dec_c)
        enc, dec = self.make_inputs(n=n, enc_c=enc_c, dec_c=dec_c)

        out = module(enc, dec)

        assert out.shape == (n, enc_c)

    def test_forward_preserves_dtype(self):
        module = GatedSkipConnectionCL(enc_channels=32, dec_channels=16)

        enc = torch.randn(10, 32, dtype=torch.float32)
        dec = torch.randn(10, 16, dtype=torch.float32)

        out = module(enc, dec)

        assert out.dtype == torch.float32

    def test_forward_returns_finite_values(self):
        module = GatedSkipConnectionCL(enc_channels=64, dec_channels=32)
        enc, dec = self.make_inputs(n=12, enc_c=64, dec_c=32)

        out = module(enc, dec)

        assert torch.isfinite(out).all()

    # =====================================================
    # Core mathematical behavior
    # =====================================================

    def test_forward_matches_exact_formula_same_channels(self):
        module = GatedSkipConnectionCL(enc_channels=64, dec_channels=64)
        enc, dec = self.make_inputs(n=10, enc_c=64, dec_c=64)

        alpha = module.gate(torch.cat([enc, dec], dim=-1))
        dec_proj = module.dec_proj(dec)
        expected = alpha * enc + (1.0 - alpha) * dec_proj

        out = module(enc, dec)

        assert torch.allclose(out, expected, atol=1e-6)

    def test_forward_matches_exact_formula_different_channels(self):
        module = GatedSkipConnectionCL(enc_channels=64, dec_channels=32)
        enc, dec = self.make_inputs(n=10, enc_c=64, dec_c=32)

        alpha = module.gate(torch.cat([enc, dec], dim=-1))
        dec_proj = module.dec_proj(dec)
        expected = alpha * enc + (1.0 - alpha) * dec_proj

        out = module(enc, dec)

        assert torch.allclose(out, expected, atol=1e-6)

    def test_alpha_values_are_in_unit_interval(self):
        module = GatedSkipConnectionCL(enc_channels=64, dec_channels=32)
        enc, dec = self.make_inputs(n=10, enc_c=64, dec_c=32)

        alpha = module.gate(torch.cat([enc, dec], dim=-1))

        assert torch.all(alpha >= 0)
        assert torch.all(alpha <= 1)

    # =====================================================
    # Golden tests with controlled alpha
    # =====================================================

    def test_when_alpha_is_one_output_equals_encoder_features(self):
        module = GatedSkipConnectionCL(enc_channels=32, dec_channels=32)
        enc, dec = self.make_inputs(n=8, enc_c=32, dec_c=32)

        class ConstantOneGate(nn.Module):
            def forward(self, x):
                return torch.ones(x.shape[0], 32, dtype=x.dtype, device=x.device)

        module.gate = ConstantOneGate()

        out = module(enc, dec)

        assert torch.allclose(out, enc, atol=1e-6)

    def test_when_alpha_is_zero_output_equals_projected_decoder_features(self):
        module = GatedSkipConnectionCL(enc_channels=64, dec_channels=32)
        enc, dec = self.make_inputs(n=8, enc_c=64, dec_c=32)

        class ConstantZeroGate(nn.Module):
            def forward(self, x):
                return torch.zeros(x.shape[0], 64, dtype=x.dtype, device=x.device)

        module.gate = ConstantZeroGate()

        out = module(enc, dec)
        expected = module.dec_proj(dec)

        assert torch.allclose(out, expected, atol=1e-6)

    def test_when_alpha_is_half_output_is_exact_average_of_enc_and_projected_dec(self):
        module = GatedSkipConnectionCL(enc_channels=64, dec_channels=32)
        enc, dec = self.make_inputs(n=8, enc_c=64, dec_c=32)

        class ConstantHalfGate(nn.Module):
            def forward(self, x):
                return torch.full((x.shape[0], 64), 0.5, dtype=x.dtype, device=x.device)

        module.gate = ConstantHalfGate()

        out = module(enc, dec)
        expected = 0.5 * enc + 0.5 * module.dec_proj(dec)

        assert torch.allclose(out, expected, atol=1e-6)

    # =====================================================
    # Projection-specific regression behavior
    # =====================================================

    def test_different_channel_sizes_work_without_runtime_error(self):
        module = GatedSkipConnectionCL(enc_channels=128, dec_channels=64)
        enc, dec = self.make_inputs(n=6, enc_c=128, dec_c=64)

        out = module(enc, dec)

        assert out.shape == (6, 128)
        assert torch.isfinite(out).all()

    def test_projection_is_identity_when_channel_sizes_match(self):
        module = GatedSkipConnectionCL(enc_channels=32, dec_channels=32)
        dec = torch.randn(7, 32)

        projected = module.dec_proj(dec)

        assert torch.allclose(projected, dec, atol=1e-7)

    def test_projection_changes_last_dimension_when_channels_differ(self):
        module = GatedSkipConnectionCL(enc_channels=48, dec_channels=24)
        dec = torch.randn(5, 24)

        projected = module.dec_proj(dec)

        assert projected.shape == (5, 48)

    # =====================================================
    # LayerNorm / stability
    # =====================================================

    def test_batch_size_one_is_supported(self):
        module = GatedSkipConnectionCL(enc_channels=64, dec_channels=32)

        enc = torch.randn(1, 64)
        dec = torch.randn(1, 32)

        out = module(enc, dec)

        assert out.shape == (1, 64)
        assert torch.isfinite(out).all()

    def test_zero_inputs_are_handled(self):
        module = GatedSkipConnectionCL(enc_channels=32, dec_channels=16)

        enc = torch.zeros(4, 32)
        dec = torch.zeros(4, 16)

        out = module(enc, dec)

        assert torch.isfinite(out).all()

    def test_large_magnitude_inputs_remain_finite(self):
        module = GatedSkipConnectionCL(enc_channels=64, dec_channels=32)

        enc = torch.randn(8, 64) * 1e6
        dec = torch.randn(8, 32) * 1e6

        out = module(enc, dec)

        assert torch.isfinite(out).all()

    # =====================================================
    # Determinism
    # =====================================================

    def test_forward_is_deterministic_for_same_inputs(self):
        module = GatedSkipConnectionCL(enc_channels=64, dec_channels=32)
        enc, dec = self.make_inputs(n=9, enc_c=64, dec_c=32)

        out1 = module(enc, dec)
        out2 = module(enc, dec)

        assert torch.equal(out1, out2)

    # =====================================================
    # Gradients
    # =====================================================

    def test_gradients_flow_to_inputs_and_parameters_same_channels(self):
        module = GatedSkipConnectionCL(enc_channels=32, dec_channels=32)

        enc = torch.randn(10, 32, requires_grad=True)
        dec = torch.randn(10, 32, requires_grad=True)

        out = module(enc, dec)
        loss = out.mean()
        loss.backward()

        assert enc.grad is not None
        assert dec.grad is not None

        for name, param in module.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"

    def test_gradients_flow_when_projection_is_used(self):
        module = GatedSkipConnectionCL(enc_channels=64, dec_channels=16)

        enc = torch.randn(10, 64, requires_grad=True)
        dec = torch.randn(10, 16, requires_grad=True)

        out = module(enc, dec)
        loss = out.pow(2).mean()
        loss.backward()

        assert enc.grad is not None
        assert dec.grad is not None
        assert module.dec_proj.weight.grad is not None
        assert module.dec_proj.bias.grad is not None

    def test_gradients_are_finite(self):
        module = GatedSkipConnectionCL(enc_channels=64, dec_channels=32)

        enc = torch.randn(7, 64, requires_grad=True)
        dec = torch.randn(7, 32, requires_grad=True)

        out = module(enc, dec)
        loss = out.sum()
        loss.backward()

        assert torch.isfinite(enc.grad).all()
        assert torch.isfinite(dec.grad).all()

        for _, param in module.named_parameters():
            assert torch.isfinite(param.grad).all()

    # =====================================================
    # Serialization
    # =====================================================

    def test_state_dict_roundtrip_preserves_behavior(self):
        module = GatedSkipConnectionCL(enc_channels=64, dec_channels=32)
        enc, dec = self.make_inputs(n=10, enc_c=64, dec_c=32)

        out1 = module(enc, dec)

        buffer = io.BytesIO()
        torch.save(module.state_dict(), buffer)
        buffer.seek(0)

        restored = GatedSkipConnectionCL(enc_channels=64, dec_channels=32)
        restored.load_state_dict(torch.load(buffer))

        out2 = restored(enc, dec)

        assert torch.allclose(out1, out2, atol=1e-7)

    # =====================================================
    # Optimizer sanity
    # =====================================================

    def test_optimizer_step_updates_parameters(self):
        module = GatedSkipConnectionCL(enc_channels=64, dec_channels=32)
        optimizer = torch.optim.Adam(module.parameters(), lr=1e-3)

        enc, dec = self.make_inputs(n=12, enc_c=64, dec_c=32)

        before = {name: p.detach().clone() for name, p in module.named_parameters()}

        out = module(enc, dec)
        loss = out.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        changed = []
        for name, p in module.named_parameters():
            changed.append(not torch.allclose(before[name], p.detach()))

        assert any(changed)
