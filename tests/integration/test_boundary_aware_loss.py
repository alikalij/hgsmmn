import torch
import torch.nn as nn
import pytest
import mock_dependencies

from pointcept.models.losses.builder import build_criteria
from pointcept.models.losses.misc import BoundaryAwareLoss
from pointcept.models.default import DefaultSegmentor, DefaultSegmentorV2


class DummyPointBackbone(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, point):
        # در Pointcept بک‌بون‌های مبتنی بر Point ویژگی feat را آپدیت می‌کنند
        point.feat = self.linear(point.feat)
        return point

# شبیه‌ساز بک‌بون برای نسخه قدیمی که مستقیم تنسور/دیکشنری را پردازش می‌کند
class DummyLogitBackbone(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_channels, num_classes)

    def forward(self, input_dict):
        return self.linear(input_dict["feat"])


@pytest.fixture
def input_dict():
    torch.manual_seed(0)
    n = 12
    num_classes = 4
    return {
        "feat": torch.randn(n, 3),
        "coord": torch.randn(n, 3),
        "offset": torch.tensor([n], dtype=torch.int32),
        "segment": torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 0, 1, 2, -1], dtype=torch.long),
        "num_classes": num_classes,
    }


@pytest.fixture
def criteria_cfg():
    return [
        dict(
            type="BoundaryAwareLoss",
            k=3,
            boundary_weight=2.0,
            ignore_index=-1,
            loss_weight=1.0,
            enable_bal=True,
        )
    ]


def test_build_criteria_constructs_boundary_aware_loss(criteria_cfg):
    """
    Registry + builder integration:
    build_criteria(...) should instantiate BoundaryAwareLoss from config.
    """
    criteria = build_criteria(criteria_cfg)

    assert len(criteria.criteria) == 1
    assert isinstance(criteria.criteria[0], BoundaryAwareLoss)
    assert criteria.criteria[0].k == 3
    assert criteria.criteria[0].boundary_weight == 2.0
    assert criteria.criteria[0].enable_bal is True


def test_criteria_forwards_input_dict_to_boundary_aware_loss(
    input_dict, criteria_cfg, monkeypatch
):
    """
    Criteria integration:
    verifies that Criteria.__call__(..., **kwargs) forwards input_dict
    into BoundaryAwareLoss.forward(...).
    """
    pred = torch.randn(
        input_dict["feat"].shape[0],
        input_dict["num_classes"],
        requires_grad=True,
    )
    target = input_dict["segment"]

    criteria = build_criteria(criteria_cfg)
    loss_module = criteria.criteria[0]

    called = {"seen": False, "coord_shape": None, "offset_shape": None}
    original_forward = loss_module.forward

    def wrapped_forward(pred, target, input_dict):
        called["seen"] = True
        called["coord_shape"] = tuple(input_dict["coord"].shape)
        called["offset_shape"] = tuple(input_dict["offset"].shape)
        return original_forward(pred, target, input_dict)

    monkeypatch.setattr(loss_module, "forward", wrapped_forward)

    loss = criteria(pred, target, input_dict=input_dict)

    assert called["seen"] is True
    assert called["coord_shape"] == tuple(input_dict["coord"].shape)
    assert called["offset_shape"] == tuple(input_dict["offset"].shape)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


from pointcept.models.default import DefaultSegmentor, DefaultSegmentorV2

def test_default_segmentor_v2_end_to_end_with_boundary_aware_loss(
    input_dict, criteria_cfg, monkeypatch
):
    """
    تست یکپارچگی DefaultSegmentorV2 با BoundaryAwareLoss
    این سگمنتور input_dict را به درستی به criteria پاس می‌دهد.
    """
    in_channels = input_dict["feat"].shape[1]
    num_classes = input_dict["num_classes"]
    backbone_out_channels = 64 # یک مقدار دلخواه برای خروجی بک‌بون

    # Mock کردن سیستم رجیستری Pointcept
    monkeypatch.setattr(
        "pointcept.models.default.build_model",
        lambda cfg: DummyPointBackbone(
            in_channels=in_channels,
            out_channels=backbone_out_channels,
        ),
    )

    # مقداردهی کلاس بر اساس امضای دقیق در default.py
    model = DefaultSegmentorV2(
        num_classes=num_classes,
        backbone_out_channels=backbone_out_channels,
        backbone={}, # یک دیکشنری خالی برای عبور از خطاها
        criteria=criteria_cfg,
    )
    model.train()

    # اجرای Forward Pass
    output = model(input_dict)

    # بررسی خروجی‌ها
    assert isinstance(output, dict)
    assert "loss" in output
    loss = output["loss"]
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    # بررسی جریان گرادیان (Backward Pass)
    loss.backward()

    for name, param in model.backbone.named_parameters():
        assert param.grad is not None, f"Missing grad for backbone parameter: {name}"
        assert torch.isfinite(param.grad).all()

    for name, param in model.seg_head.named_parameters():
        assert param.grad is not None, f"Missing grad for seg_head parameter: {name}"


def test_default_segmentor_is_incompatible_with_boundary_aware_loss_as_written(
    input_dict, criteria_cfg, monkeypatch
):
    """
    تست مستندسازی ناسازگاری: DefaultSegmentor نسخه قدیمی
    آرگومان input_dict را به criteria پاس نمی‌دهد.
    """
    monkeypatch.setattr(
        "pointcept.models.default.build_model",
        lambda cfg: DummyLogitBackbone(
            in_channels=input_dict["feat"].shape[1],
            num_classes=input_dict["num_classes"],
        ),
    )

    model = DefaultSegmentor(
        backbone={},
        criteria=criteria_cfg,
    )
    model.train()

    # خطای TypeError به دلیل فراخوانی loss بدون input_dict (در BoundaryAwareLoss) رخ می‌دهد
    with pytest.raises(TypeError):
        model(input_dict)
