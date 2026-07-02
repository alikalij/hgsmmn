# file: tests/unit/test_gated_skip_connectionGE.py
import io
import copy
import pytest
import torch
import torch.nn as nn
from unittest.mock import patch
import mock_dependencies

from pointcept.models.point_transformer_v3.new_modules import GatedSkipConnectionGE


class TestGatedSkipConnectionGE:
    """Independent and comprehensive test suite for GatedSkipConnectionGE."""

    def make_inputs(self, n=16, enc_c=64, dec_c=64, seed=123):
        torch.manual_seed(seed)
        encoder = torch.randn(n, enc_c)
        decoder = torch.randn(n, dec_c)
        return encoder, decoder

    # =====================================================
    # 1. Initialization Tests
    # =====================================================

    def test_init_builds_expected_submodules(self):
        module = GatedSkipConnectionGE(64, 128)
        assert isinstance(module.W_g, nn.Linear)
        assert isinstance(module.W_x, nn.Linear)
        assert isinstance(module.psi, nn.Sequential)
        assert module.W_g.in_features == 128
        assert module.W_x.in_features == 64

    @pytest.mark.parametrize("enc, dec, expected_inter", [
        (64, 128, 32),
        (128, 64, 32),
        (100, 100, 50),
        (1, 1, 1), # Edge case: max(1//2, 1) = 1
    ])
    def test_default_inter_channels_logic(self, enc, dec, expected_inter):
        module = GatedSkipConnectionGE(enc, dec)
        assert module.W_g.out_features == expected_inter
        assert module.W_x.out_features == expected_inter

    def test_custom_inter_channels(self):
        module = GatedSkipConnectionGE(64, 128, inter_channels=48)
        assert module.W_g.out_features == 48

    # =====================================================
    # 2. Forward Pass & Shape Tests
    # =====================================================

    @pytest.mark.parametrize("batch_size", [1, 5, 100])
    def test_forward_output_shape_and_batch_sizes(self, batch_size):
        module = GatedSkipConnectionGE(32, 32)
        enc, dec = self.make_inputs(n=batch_size, enc_c=32, dec_c=32)
        out = module(enc, dec)
        assert out.shape == (batch_size, 32)

    def test_forward_with_channel_mismatch(self):
        module = GatedSkipConnectionGE(64, 128)
        enc, dec = self.make_inputs(n=10, enc_c=64, dec_c=128)
        out = module(enc, dec)
        assert out.shape == (10, 64) # Output shape must match encoder

    # =====================================================
    # 3. Core Mathematical Logic
    # =====================================================

    def test_forward_matches_exact_formula(self):
        module = GatedSkipConnectionGE(64, 64)
        enc, dec = self.make_inputs(n=10, enc_c=64, dec_c=64)
        
        with torch.no_grad():
            g = module.W_g(dec)
            x = module.W_x(enc)
            psi = module.psi(g + x)
            expected = (enc * psi) + dec
            
        out = module(enc, dec)
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-7)

    def test_psi_range_is_valid(self):
        module = GatedSkipConnectionGE(64, 64)
        enc, dec = self.make_inputs(n=50, enc_c=64, dec_c=64)
        out = module(enc, dec)
        
        # Manually extract psi
        psi = module.psi(module.W_g(dec) + module.W_x(enc))
        assert torch.all(psi >= 0.0)
        assert torch.all(psi <= 1.0)

    # =====================================================
    # 4. Golden Behavior Tests (Mocking Psi)
    # =====================================================

    def test_psi_zero_returns_decoder(self):
        module = GatedSkipConnectionGE(32, 32)
        enc, dec = self.make_inputs(n=10, enc_c=32, dec_c=32)

        with patch.object(module.psi, 'forward', return_value=torch.zeros(10, 1)):
            out = module(enc, dec)
            torch.testing.assert_close(out, dec)

    def test_psi_one_returns_sum(self):
        module = GatedSkipConnectionGE(32, 32)
        enc, dec = self.make_inputs(n=10, enc_c=32, dec_c=32)

        with patch.object(module.psi, 'forward', return_value=torch.ones(10, 1)):
            out = module(enc, dec)
            torch.testing.assert_close(out, enc + dec)

    def test_psi_half_interpolation(self):
        module = GatedSkipConnectionGE(32, 32)
        enc, dec = self.make_inputs(n=10, enc_c=32, dec_c=32)

        with patch.object(module.psi, 'forward', return_value=torch.full((10, 1), 0.5)):
            out = module(enc, dec)
            torch.testing.assert_close(out, (0.5 * enc) + dec)

    # =====================================================
    # 5. Regression & Edge Cases
    # =====================================================

    def test_psi_broadcasting_correctness(self):
        module = GatedSkipConnectionGE(64, 64)
        enc, dec = self.make_inputs(n=10, enc_c=64, dec_c=64)

        with patch.object(module.psi, 'forward', return_value=torch.ones(10, 1) * 2.0):
            out = module(enc, dec)
            torch.testing.assert_close(out, (enc * 2.0) + dec)

    def test_zero_and_large_inputs(self):
        module = GatedSkipConnectionGE(64, 64)
        # Zeros
        out_zero = module(torch.zeros(5, 64), torch.zeros(5, 64))
        assert not torch.isnan(out_zero).any()
        # Large values
        out_large = module(torch.randn(5, 64) * 1e6, torch.randn(5, 64) * 1e6)
        assert not torch.isnan(out_large).any()

    # =====================================================
    # 6. Gradient Flow Tests
    # =====================================================

    def test_gradients_flow_to_inputs_and_parameters(self):
        module = GatedSkipConnectionGE(32, 32)
        enc = torch.randn(10, 32, requires_grad=True)
        dec = torch.randn(10, 32, requires_grad=True)
        
        out = module(enc, dec)
        loss = out.sum()
        loss.backward()
        
        assert enc.grad is not None and not torch.all(enc.grad == 0)
        assert dec.grad is not None and not torch.all(dec.grad == 0)
        
        for name, param in module.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.all(param.grad == 0)

    # =====================================================
    # 7. Determinism & Serialization
    # =====================================================

    def test_forward_is_deterministic(self):
        module = GatedSkipConnectionGE(64, 64)
        module.eval()
        enc, dec = self.make_inputs()
        
        out1 = module(enc, dec)
        out2 = module(enc, dec)
        torch.testing.assert_close(out1, out2)

    def test_state_dict_roundtrip(self):
        module1 = GatedSkipConnectionGE(64, 64)
        enc, dec = self.make_inputs()
        out1 = module1(enc, dec)
        
        module2 = GatedSkipConnectionGE(64, 64)
        module2.load_state_dict(copy.deepcopy(module1.state_dict()))
        out2 = module2(enc, dec)
        
        torch.testing.assert_close(out1, out2)

    # =====================================================
    # 8. Integration / Real-world Scenario
    # =====================================================

    def test_training_step_updates_weights(self):
        module = GatedSkipConnectionGE(64, 64)
        optimizer = torch.optim.Adam(module.parameters(), lr=0.01)
        enc, dec = self.make_inputs()
        target = torch.randn(16, 64)
        
        before = {name: p.clone().detach() for name, p in module.named_parameters()}
        
        out = module(enc, dec)
        loss = nn.MSELoss()(out, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        changed = any(not torch.allclose(before[name], p) for name, p in module.named_parameters())
        assert changed, "Parameters did not update after optimizer step"
