# file: tests/unit/test_boundary_aware_loss.py
import sys
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock
import mock_dependencies

from pointcept.models.losses.misc import BoundaryAwareLoss

# ==========================================
# 1. Mocking External CUDA Dependencies
# ==========================================
# قبل از ایمپورت یا تعریف کلاس، ماژول pointops را به طور کامل Mock می‌کنیم
# تا در محیط‌هایی که CUDA یا این کتابخانه نصب نیست، تست‌ها با خطا مواجه نشوند.
mock_pointops = MagicMock()
sys.modules['pointops'] = mock_pointops

@pytest.fixture(autouse=True)
def reset_pointops_mock():
    """Reset mock state before each test to prevent leakage."""
    mock_pointops.reset_mock()
    mock_pointops.knn_query.reset_mock()
    mock_pointops.knn_query.return_value = None
    yield

# ==========================================
# 3. Test Suites (10 Categories)
# ==========================================

class TestBoundaryAwareLoss:
    def test_default_parameters(self):
        loss = BoundaryAwareLoss()
        assert loss.k == 8
        assert loss.boundary_weight == 2.0
        assert loss.ignore_index == -1
        assert loss.loss_weight == 1.0
        assert loss.enable_bal is True
        assert loss.ce.reduction == 'none'

    def test_forward_with_enable_bal_false(self):
        N, C = 50, 5
        loss_fn = BoundaryAwareLoss(enable_bal=False)
        pred = torch.randn(N, C)
        target = torch.randint(0, C, (N,))
        input_dict = {"coord": torch.randn(N, 3), "offset": torch.tensor([N], dtype=torch.int32)}
        
        mock_pointops.knn_query.reset_mock()
        loss = loss_fn(pred, target, input_dict)
        
        # بررسی اینکه تابع KNN اصلاً صدا زده نشده است (اطمینان از رفع باگ)
        mock_pointops.knn_query.assert_not_called()
        assert loss.item() >= 0

    def test_compute_boundary_mask_exact_math(self):
        loss_fn = BoundaryAwareLoss(k=2, boundary_weight=3.5)
        # 5 نقطه: نقطه 0 مرز نیست، نقطه 1 مرز است، نقطه 4 اینگنور است
        target = torch.tensor([0, 0, 1, 1, -1], dtype=torch.long)
        coord = torch.randn(5, 3)
        offset = torch.tensor([5], dtype=torch.int32)
        
        # شبیه‌سازی دقیق ایندکس‌های همسایگان برای اعتبارسنجی منطق
        mock_idx = torch.tensor([
            [0, 1], # labels: 0, 0 -> Not Boundary
            [0, 2], # labels: 0, 1 -> Boundary!
            [2, 3], # labels: 1, 1 -> Not Boundary
            [2, 4], # labels: 1, -1 -> Ignore index neighbor shouldn't trigger boundary
            [4, 4]  # center is ignore index
        ])
        mock_pointops.knn_query.return_value = (mock_idx, None)
        
        weights = loss_fn.compute_boundary_mask(target, coord, offset)
        
        # فقط نقطه دوم (ایندکس 1) باید وزن مرزی 3.5 بگیرد
        expected_weights = torch.tensor([1.0, 3.5, 1.0, 1.0, 1.0])
        assert torch.allclose(weights, expected_weights)

    def test_all_ignored_points_prevents_nan(self):
        N, C = 10, 5
        loss_fn = BoundaryAwareLoss(ignore_index=-1)
        pred = torch.randn(N, C)
        target = torch.full((N,), -1, dtype=torch.long) # همه ignore
        input_dict = {"coord": torch.randn(N, 3), "offset": torch.tensor([N], dtype=torch.int32)}
        
        loss = loss_fn(pred, target, input_dict)
        
        # چک کنید که KNN اصلاً صدا زده نشده (چون همه ignore هستند)
        mock_pointops.knn_query.assert_not_called()
        assert not torch.isnan(loss), "Loss should not be NaN when all targets are ignored"
        assert loss.item() == 0.0

    def test_loss_weight_scaling(self):
        N, C = 10, 5
        pred = torch.randn(N, C)
        target = torch.randint(0, C, (N,))
        input_dict = {"coord": torch.randn(N, 3), "offset": torch.tensor([N], dtype=torch.int32)}
        
        mock_pointops.knn_query.return_value = (torch.zeros(N, 8), None)
        
        loss_1 = BoundaryAwareLoss(loss_weight=1.0)(pred, target, input_dict)
        loss_2 = BoundaryAwareLoss(loss_weight=2.0)(pred, target, input_dict)
        
        assert torch.allclose(loss_2, loss_1 * 2.0, rtol=1e-5)

    def test_gradient_flow_through_loss(self):
        N, C = 20, 5
        loss_fn = BoundaryAwareLoss()
        pred = torch.randn(N, C, requires_grad=True)
        target = torch.randint(0, C, (N,))
        input_dict = {"coord": torch.randn(N, 3), "offset": torch.tensor([N], dtype=torch.int32)}
        
        mock_pointops.knn_query.return_value = (torch.zeros(N, 8), None)
        
        loss = loss_fn(pred, target, input_dict)
        loss.backward()
        
        assert pred.grad is not None
        assert not torch.isnan(pred.grad).any()
        assert (pred.grad.abs() > 0).any()

    def test_single_point(self):
        N, C = 1, 5
        loss_fn = BoundaryAwareLoss(k=1)
        pred = torch.randn(N, C)
        target = torch.tensor([0])
        input_dict = {"coord": torch.randn(N, 3), "offset": torch.tensor([N], dtype=torch.int32)}
        
        mock_pointops.knn_query.return_value = (torch.zeros(1, 1), None)
        loss = loss_fn(pred, target, input_dict)
        
        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_deterministic_output(self):
        N, C = 20, 5
        pred = torch.randn(N, C)
        target = torch.randint(0, C, (N,))
        input_dict = {"coord": torch.randn(N, 3), "offset": torch.tensor([N], dtype=torch.int32)}
        
        mock_pointops.knn_query.return_value = (torch.zeros(N, 8), None)
        loss_fn = BoundaryAwareLoss()
        
        torch.manual_seed(42)
        loss_1 = loss_fn(pred, target, input_dict)
        
        torch.manual_seed(42)
        loss_2 = loss_fn(pred, target, input_dict)
        
        assert torch.allclose(loss_1, loss_2)

    def test_extreme_logits(self):
        N, C = 20, 5
        loss_fn = BoundaryAwareLoss()
        pred_large = torch.randn(N, C) * 100 # Logits بسیار بزرگ
        target = torch.randint(0, C, (N,))
        input_dict = {"coord": torch.randn(N, 3), "offset": torch.tensor([N], dtype=torch.int32)}
        
        mock_pointops.knn_query.return_value = (torch.zeros(N, 8), None)
        loss = loss_fn(pred_large, target, input_dict)
        
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_training_loop_simulation(self):
        N, C = 50, 5
        loss_fn = BoundaryAwareLoss()
        model = nn.Linear(3, C)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        coord = torch.randn(N, 3)
        target = torch.randint(0, C, (N,))
        input_dict = {"coord": coord, "offset": torch.tensor([N], dtype=torch.int32)}
        
        mock_pointops.knn_query.return_value = (torch.zeros(N, 8), None)
        
        initial_loss = None
        for _ in range(3):
            optimizer.zero_grad()
            pred = model(coord)
            loss = loss_fn(pred, target, input_dict)
            
            if initial_loss is None:
                initial_loss = loss.item()
                
            loss.backward()
            optimizer.step()
            
        assert not torch.isnan(loss)
        assert loss.item() != initial_loss # بررسی اینکه وزن‌ها آپدیت می‌شوند
