# file: tests/unit/test_serialization_pe.py
import mock_dependencies
import pytest
import torch
import torch.nn as nn

# --- Import ---
# این بخش عالی است و نیازی به تغییر ندارد.
from pointcept.models.point_transformer_v3.new_modules import SerializationPositionalEncoding


# --- Helper Function ---
# این بخش هم عالی است.
def get_trainable_parameter_grads(module: nn.Module):
    """Return gradients of trainable parameters that received gradients."""
    return [p.grad for p in module.parameters() if p.requires_grad and p.grad is not None]


# --- Fixtures (با اصلاح dtype) ---

@pytest.fixture
def sample_features():
    """Sample feature tensor of shape (N, C)."""
    torch.manual_seed(0)
    return torch.randn(128, 64)


@pytest.fixture
def sample_serialized_order():
    """
    Sample non-trivial serialized order of shape (N,).
    
    نکته کلیدی: نوع داده صحیح برای ترتیب، torch.long است.
    """
    torch.manual_seed(0)
    # FIX: Changed from .float() to .long() for semantic correctness.
    return torch.randperm(128, dtype=torch.long)


@pytest.fixture
def single_point_features():
    """Features for a single point."""
    torch.manual_seed(0)
    return torch.randn(1, 64)


@pytest.fixture
def single_point_order():
    """Serialized order for a single point."""
    # FIX: Changed from tensor([0.0]) to tensor([0], dtype=torch.long)
    return torch.tensor([0], dtype=torch.long)


@pytest.fixture
def spe_instance_default():
    """SPE instance with default hidden_dim."""
    return SerializationPositionalEncoding(channels=64)


@pytest.fixture
def spe_instance_custom():
    """SPE instance with custom hidden_dim."""
    return SerializationPositionalEncoding(channels=128, hidden_dim=32)


# --- Test Class ---

class TestSerializationPositionalEncoding:
    """Unit tests for SerializationPositionalEncoding."""

    # تست‌های Instantiation شما عالی و کامل هستند. بدون تغییر.
    @pytest.mark.sanity
    def test_instantiation_default(self, spe_instance_default):
        """Default hidden_dim should match the real implementation."""
        assert isinstance(spe_instance_default, nn.Module)
        assert spe_instance_default.channels == 64
        assert spe_instance_default.hidden_dim == 16
        assert spe_instance_default.mlp[0].in_features == 1
        assert spe_instance_default.mlp[0].out_features == 16
        assert isinstance(spe_instance_default.mlp[1], nn.ReLU)
        assert spe_instance_default.mlp[2].in_features == 16
        assert spe_instance_default.mlp[2].out_features == 64

    @pytest.mark.sanity
    def test_instantiation_custom(self, spe_instance_custom):
        """Custom hidden_dim should be reflected in MLP dimensions."""
        assert isinstance(spe_instance_custom, nn.Module)
        assert spe_instance_custom.channels == 128
        assert spe_instance_custom.hidden_dim == 32
        assert spe_instance_custom.mlp[0].out_features == 32
        assert spe_instance_custom.mlp[2].in_features == 32
        assert spe_instance_custom.mlp[2].out_features == 128

    @pytest.mark.parametrize(
        "channels, hidden_dim",
        [(16, 4), (32, 8), (64, 16), (128, 32), (256, 64)],
    )
    def test_instantiation_various_configs(self, channels, hidden_dim):
        """Various constructor configs should build the expected MLP."""
        module = SerializationPositionalEncoding(channels=channels, hidden_dim=hidden_dim)
        assert isinstance(module, nn.Module)
        assert module.channels == channels
        assert module.hidden_dim == hidden_dim
        assert module.mlp[0].in_features == 1
        assert module.mlp[0].out_features == hidden_dim
        assert module.mlp[2].in_features == hidden_dim
        assert module.mlp[2].out_features == channels

    # --- تست‌های Forward Pass (با اصلاح dtype) ---
    @pytest.mark.sanity
    @pytest.mark.parametrize("N, C", [(1, 64), (10, 32), (128, 64), (500, 128)])
    def test_forward_preserves_shape_dtype_and_device(self, N, C):
        """Output should preserve input feature shape, dtype, and device."""
        torch.manual_seed(0)
        module = SerializationPositionalEncoding(channels=C)
        features = torch.randn(N, C, dtype=torch.float32)
        # FIX: Using semantically correct long type for order.
        order = torch.randperm(N, dtype=torch.long)

        output = module(features, order)

        assert output.shape == features.shape
        assert output.dtype == features.dtype
        assert output.device == features.device

    # تست‌های منطقی شما عالی هستند. بدون تغییر.
    @pytest.mark.logic
    def test_forward_changes_features(self, sample_features, sample_serialized_order):
        """Output should usually differ from input due to added positional encoding."""
        torch.manual_seed(0)
        module = SerializationPositionalEncoding(channels=sample_features.shape[1])
        features_original = sample_features.clone()

        output = module(sample_features, sample_serialized_order)

        assert not torch.allclose(output, features_original, atol=1e-6)

    @pytest.mark.logic
    def test_forward_does_not_modify_input_inplace(self, sample_features, sample_serialized_order):
        """Input feature tensor should remain unchanged after forward."""
        torch.manual_seed(0)
        module = SerializationPositionalEncoding(channels=sample_features.shape[1])
        features_original = sample_features.clone()

        _ = module(sample_features, sample_serialized_order)

        assert torch.allclose(sample_features, features_original, atol=1e-6)

    # این تست‌های White-box پیشرفته، عالی هستند و حفظ می‌شوند.
    @pytest.mark.logic
    def test_normalized_order_is_unsqueezed_and_bounded(self, sample_features, sample_serialized_order):
        """
        Capture the first Linear input to verify:
        - normalized order shape is (N, 1)
        - values are in [0, 1]
        """
        torch.manual_seed(0)
        module = SerializationPositionalEncoding(channels=sample_features.shape[1])
        captured = {}

        def hook(_module, inputs, _output):
            captured["linear_input"] = inputs[0].detach().clone()

        handle = module.mlp[0].register_forward_hook(hook)
        try:
            _ = module(sample_features, sample_serialized_order)
        finally:
            handle.remove()

        assert "linear_input" in captured
        linear_input = captured["linear_input"]
        assert linear_input.shape == (sample_features.shape[0], 1)
        assert torch.all(linear_input >= 0.0)
        assert torch.all(linear_input <= 1.0)

    @pytest.mark.edge_case
    def test_all_same_order_maps_to_zero_normalized_input(self, sample_features):
        """
        When all serialized_order values are equal, implementation should avoid division
        by zero and feed zeros into the MLP.
        """
        torch.manual_seed(0)
        module = SerializationPositionalEncoding(channels=sample_features.shape[1])
        same_order = torch.zeros(sample_features.shape[0], dtype=torch.long) # Using long
        captured = {}

        def hook(_module, inputs, _output):
            captured["linear_input"] = inputs[0].detach().clone()

        handle = module.mlp[0].register_forward_hook(hook)
        try:
            output = module(sample_features, same_order)
        finally:
            handle.remove()

        assert "linear_input" in captured
        linear_input = captured["linear_input"]
        assert linear_input.shape == (sample_features.shape[0], 1)
        assert torch.allclose(linear_input, torch.zeros_like(linear_input), atol=1e-7)
        assert torch.isfinite(output).all()

    @pytest.mark.logic
    def test_order_sensitivity(self, sample_features):
        """Different serialized orders should produce different outputs."""
        torch.manual_seed(0)
        module = SerializationPositionalEncoding(channels=sample_features.shape[1])
        n = sample_features.shape[0]
        # Using long as the primary type
        order_a = torch.arange(n, dtype=torch.long)
        order_b = torch.arange(n - 1, -1, -1, dtype=torch.long)

        out_a = module(sample_features, order_a)
        out_b = module(sample_features, order_b)

        assert not torch.allclose(out_a, out_b, atol=1e-6)

    # این تست سازگاری dtype بسیار عالی است و باید بماند.
    @pytest.mark.logic
    def test_order_dtype_handling_float_and_long_match(self, sample_features):
        """Same order values with different dtypes should produce identical outputs."""
        torch.manual_seed(0)
        module = SerializationPositionalEncoding(channels=sample_features.shape[1])
        base_order = torch.randperm(sample_features.shape[0])
        order_float = base_order.float()
        order_long = base_order.long()

        out_float = module(sample_features, order_float)
        out_long = module(sample_features, order_long)

        assert torch.allclose(out_float, out_long, atol=1e-6)

    @pytest.mark.edge_case
    def test_single_point_input(self, single_point_features, single_point_order):
        """Single-point input should work and preserve shape."""
        torch.manual_seed(0)
        module = SerializationPositionalEncoding(channels=single_point_features.shape[1])
        output = module(single_point_features, single_point_order)

        assert output.shape == single_point_features.shape
        assert output.dtype == single_point_features.dtype
        assert torch.isfinite(output).all()

    # این تست white-box هم عالی است.
    @pytest.mark.logic
    def test_residual_connection_matches_features_plus_mlp_output(self, sample_features, sample_serialized_order):
        """Forward output should equal features + mlp(normalized_order)."""
        torch.manual_seed(0)
        module = SerializationPositionalEncoding(channels=sample_features.shape[1])
        order_float = sample_serialized_order.float() # Convert to float for math
        order_min, order_max = torch.min(order_float), torch.max(order_float)
        
        if order_max == order_min:
            order_norm = torch.zeros_like(order_float)
        else:
            order_norm = (order_float - order_min) / (order_max - order_min)

        expected_pe = module.mlp(order_norm.unsqueeze(1))
        expected_output = sample_features + expected_pe
        actual_output = module(sample_features, sample_serialized_order)

        assert torch.allclose(actual_output, expected_output, atol=1e-6)

    # --- تست‌های گرادیان (عالی و بدون تغییر) ---
    @pytest.mark.gradients
    def test_gradients_flow_to_parameters(self, sample_features, sample_serialized_order):
        """Backward pass should produce gradients for trainable parameters."""
        torch.manual_seed(0)
        module = SerializationPositionalEncoding(channels=sample_features.shape[1])
        features = sample_features.clone().requires_grad_(True)
        output = module(features, sample_serialized_order)
        loss = output.sum()
        loss.backward()

        grads = get_trainable_parameter_grads(module)
        assert len(grads) > 0
        assert all(g is not None for g in grads)
        assert all(torch.isfinite(g).all() for g in grads)

    @pytest.mark.gradients
    def test_gradients_flow_to_input_features(self, sample_features, sample_serialized_order):
        """Backward pass should also propagate gradients to input features."""
        torch.manual_seed(0)
        module = SerializationPositionalEncoding(channels=sample_features.shape[1])
        features = sample_features.clone().requires_grad_(True)
        output = module(features, sample_serialized_order)
        loss = output.mean()
        loss.backward()

        assert features.grad is not None
        assert features.grad.shape == features.shape
        assert torch.isfinite(features.grad).all()

    # تست assertion هم عالی است.
    @pytest.mark.edge_case
    def test_assertion_on_length_mismatch(self):
        """Forward should assert when features and order lengths differ."""
        torch.manual_seed(0)
        module = SerializationPositionalEncoding(channels=64)
        features = torch.randn(10, 64)
        order = torch.arange(9, dtype=torch.long) # using long

        with pytest.raises(AssertionError):
            module(features, order)

    # --- تست‌های جدید پیشنهادی ---

    @pytest.mark.logic
    def test_repeatability_in_eval_mode(self, sample_features, sample_serialized_order):
        """Output should be deterministic in eval mode for the same input."""
        module = SerializationPositionalEncoding(channels=sample_features.shape[1])
        module.eval()  # Set to evaluation mode

        output1 = module(sample_features, sample_serialized_order)
        output2 = module(sample_features, sample_serialized_order)

        assert torch.allclose(output1, output2, atol=1e-7)

    @pytest.mark.edge_case
    def test_forward_output_is_finite(self, sample_features, sample_serialized_order):
        """Forward pass output must not contain NaNs or Infs."""
        module = SerializationPositionalEncoding(channels=sample_features.shape[1])
        output = module(sample_features, sample_serialized_order)
        assert torch.isfinite(output).all()

    @pytest.mark.edge_case
    def test_zero_features_input(self, sample_serialized_order):
        """Forward pass should work with zero-valued features and produce non-zero output."""
        num_features, channels = sample_serialized_order.shape[0], 64
        module = SerializationPositionalEncoding(channels=channels)
        features = torch.zeros(num_features, channels)

        output = module(features, sample_serialized_order)

        assert output.shape == features.shape
        assert torch.isfinite(output).all()
        # The output should not be zero because the positional encoding is added
        assert not torch.allclose(output, torch.zeros_like(output))

    @pytest.mark.edge_case
    def test_negative_and_positive_order_values(self):
        """
        تست می‌کند که نرمال‌سازی با مقادیر order منفی و مثبت به درستی کار می‌کند.
        این یک edge case مهم است.
        """
        module = SerializationPositionalEncoding(channels=8)
        features = torch.randn(10, 8)
        # مقادیر order شامل منفی، صفر و مثبت
        order = torch.tensor([-5, -4, -2, 0, 1, 5, 8, 10, 11, 20], dtype=torch.float32)

        output = module(features, order)

        assert output.shape == features.shape
        assert torch.isfinite(output).all()
        # خروجی نباید با ورودی یکسان باشد
        assert not torch.allclose(output, features)

    @pytest.mark.gradients
    def test_gradient_magnitude_is_reasonable(self):
        """
        تست می‌کند که اندازه گرادیان‌ها در محدوده معقولی قرار دارد
        و دچار انفجار (exploding) یا محو شدن (vanishing) نمی‌شوند.
        """
        torch.manual_seed(42)
        module = SerializationPositionalEncoding(channels=16)
        features = torch.randn(100, 16, requires_grad=True)
        order = torch.randperm(100).long()

        output = module(features, order)
        loss = output.mean()  # A reasonable loss function
        loss.backward()

        # بررسی گرادیان پارامترها
        assert len(list(module.parameters())) > 0, "Module has no parameters"
        for name, param in module.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                assert 1e-4 < grad_norm < 1e2, (
                    f"Parameter '{name}' gradient norm {grad_norm:.4f} is unreasonable. "
                    "Possible exploding/vanishing gradients."
                )

        # بررسی گرادیان ورودی
        assert features.grad is not None
        input_grad_norm = features.grad.norm().item()
        assert 1e-4 < input_grad_norm < 1e2, (
            f"Input feature gradient norm {input_grad_norm:.4f} is unreasonable."
        )

    @pytest.mark.logic
    def test_feature_permutation_invariance_with_order(self):
        """
        یک تست منطقی قدرتمند: اگر فیچرها و order ها را با هم جابجا کنیم،
        خروجی نیز باید به همان شکل جابجا شود. این نشان می‌دهد که نگاشت
        بین فیچر و موقعیت آن به درستی حفظ می‌شود.
        """
        torch.manual_seed(123)
        module = SerializationPositionalEncoding(channels=8)
        features = torch.randn(20, 8)
        order = torch.arange(20, dtype=torch.long)

        # خروجی اصلی
        output_original = module(features, order)

        # ایجاد یک جایگشت (permutation)
        perm = torch.randperm(20)
        features_permuted = features[perm]
        order_permuted = order[perm]

        # خروجی با ورودی‌های جابجا شده
        output_from_permuted = module(features_permuted, order_permuted)

        # خروجی جابجا شده باید با خروجی اصلی که جابجا شده، یکسان باشد
        assert torch.allclose(output_original[perm], output_from_permuted, atol=1e-6)

    @pytest.mark.sanity
    def test_instantiation_with_equal_channels_and_hidden_dim(self):
        """یک sanity check ساده برای حالتی که ابعاد کانال و لایه مخفی برابرند."""
        channels = 32
        module = SerializationPositionalEncoding(channels=channels, hidden_dim=channels)
        features = torch.randn(10, channels)
        order = torch.arange(10).long()

        output = module(features, order)

        assert output.shape == features.shape
        assert torch.isfinite(output).all()
