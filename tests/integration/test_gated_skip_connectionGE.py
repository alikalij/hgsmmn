import copy
from unittest.mock import patch

import pytest
import torch

from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import (
    SerializedUnpooling,
)
from pointcept.models.utils.structure import Point
from pointcept.models.point_transformer_v3.new_modules import GatedSkipConnectionGE


class SerializedUnpoolingWithGE(SerializedUnpooling):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer=None,
        act_layer=None,
        traceable=False,
        enable_gsc=False,
    ):
        super().__init__(
            in_channels=in_channels,
            skip_channels=skip_channels,
            out_channels=out_channels,
            norm_layer=norm_layer,
            act_layer=act_layer,
            traceable=traceable,
            enable_gsc=False,  # prevent parent from wiring CL
        )
        self.enable_gsc = enable_gsc
        if self.enable_gsc:
            self.gsc = GatedSkipConnectionGE(
                encoder_channels=out_channels,
                decoder_channels=out_channels,
            )


@pytest.fixture
def cpu_device():
    return torch.device("cpu")


@pytest.fixture
def point_factory(cpu_device):
    def _make_point(num_child=5, num_parent=8, in_channels=4, skip_channels=6):
        child = Point(
            {
                "feat": torch.randn(num_child, in_channels, device=cpu_device),
            }
        )
        parent = Point(
            {
                "feat": torch.randn(num_parent, skip_channels, device=cpu_device),
            }
        )
        inverse = torch.randint(
            low=0, high=num_child, size=(num_parent,), device=cpu_device
        )

        child["pooling_parent"] = parent
        child["pooling_inverse"] = inverse
        return child

    return _make_point


@pytest.fixture
def unpool_config():
    return dict(
        in_channels=4,
        skip_channels=6,
        out_channels=8,
        norm_layer=None,
        act_layer=None,
        traceable=False,
    )


class TestGatedSkipConnectionGE:
    def test_constructs_ge_when_enabled(self, unpool_config):
        layer = SerializedUnpoolingWithGE(**unpool_config, enable_gsc=True)

        assert layer.enable_gsc is True
        assert hasattr(layer, "gsc")
        assert isinstance(layer.gsc, GatedSkipConnectionGE)

    def test_has_no_ge_when_disabled(self, unpool_config):
        layer = SerializedUnpoolingWithGE(**unpool_config, enable_gsc=False)

        assert layer.enable_gsc is False
        assert not hasattr(layer, "gsc")

    def test_forward_invokes_ge_when_enabled(self, point_factory, unpool_config):
        point = point_factory()
        layer = SerializedUnpoolingWithGE(**unpool_config, enable_gsc=True)

        with patch.object(layer.gsc, "forward", wraps=layer.gsc.forward) as mock_forward:
            out = layer(copy.deepcopy(point))

        assert mock_forward.call_count == 1
        assert out.feat.shape[0] == point["pooling_parent"].feat.shape[0]

    def test_output_shape_matches_parent_cardinality(self, point_factory, unpool_config):
        point = point_factory(num_child=4, num_parent=9)
        layer = SerializedUnpoolingWithGE(**unpool_config, enable_gsc=True)

        out = layer(copy.deepcopy(point))

        assert out.feat.shape == (9, unpool_config["out_channels"])

    def test_forward_with_ge_has_grad_flow(self, point_factory, unpool_config):
        # ۱. ساخت شیء با استفاده از فیکسچر
        point_in = point_factory()
        
        # ۲. فعال کردن گرادیان‌ها
        point_in.feat.requires_grad_(True)
        point_in["pooling_parent"].feat.requires_grad_(True)

        # ۳. ذخیره رفرنس تنسورها قبل از اجرای لایه (برای جلوگیری از خطای pop)
        child_feat = point_in.feat
        parent_feat = point_in["pooling_parent"].feat

        # ۴. ساخت لایه و اجرای forward (بدون deepcopy)
        layer = SerializedUnpoolingWithGE(**unpool_config, enable_gsc=True)
        out = layer(point_in)
        
        # ۵. محاسبه Loss و Backward
        loss = out.feat.sum()
        loss.backward()

        # ۶. بررسی گرادیان‌های ورودی
        assert child_feat.grad is not None, "Child feature should receive gradients."
        assert parent_feat.grad is not None, "Parent feature should receive gradients."
        
        # ۷. بررسی گرادیان پارامترهای خود لایه GE (اطمینان از متصل بودن GE به گراف)
        assert any(p.grad is not None for p in layer.gsc.parameters()), "GE parameters should receive gradients."

    def test_traceable_attaches_unpooling_parent(self, point_factory, unpool_config):
        point = point_factory()
        config = {**unpool_config, "traceable": True}
        layer = SerializedUnpoolingWithGE(**config, enable_gsc=True)

        out = layer(copy.deepcopy(point))

        assert "unpooling_parent" in out
        assert isinstance(out["unpooling_parent"], Point)
        assert out["unpooling_parent"].feat.shape[1] == unpool_config["out_channels"]

    def test_forward_without_ge_uses_baseline_addition(
        self, point_factory, unpool_config
    ):
        point = point_factory()
        layer = SerializedUnpoolingWithGE(**unpool_config, enable_gsc=False)

        out = layer(copy.deepcopy(point))

        parent = point["pooling_parent"]
        inverse = point["pooling_inverse"]

        projected_child = layer.proj(copy.deepcopy(point))
        projected_parent = layer.proj_skip(copy.deepcopy(parent))
        expected = projected_parent.feat + projected_child.feat[inverse]

        assert torch.allclose(out.feat, expected, atol=1e-6)

    def test_ge_path_differs_from_plain_addition(self, point_factory, unpool_config):
        point = point_factory()

        layer_ge = SerializedUnpoolingWithGE(**unpool_config, enable_gsc=True)
        layer_plain = SerializedUnpoolingWithGE(**unpool_config, enable_gsc=False)

        out_ge = layer_ge(copy.deepcopy(point))
        out_plain = layer_plain(copy.deepcopy(point))

        assert not torch.allclose(out_ge.feat, out_plain.feat)

    def test_ge_matches_manual_formula(self, point_factory, unpool_config):
        point = point_factory(num_child=6, num_parent=7)
        layer = SerializedUnpoolingWithGE(**unpool_config, enable_gsc=True)

        out = layer(copy.deepcopy(point))

        parent = point["pooling_parent"]
        inverse = point["pooling_inverse"]

        projected_child = layer.proj(copy.deepcopy(point))
        projected_parent = layer.proj_skip(copy.deepcopy(parent))
        decoder_upsampled = projected_child.feat[inverse]

        expected = layer.gsc(projected_parent.feat, decoder_upsampled)

        assert torch.allclose(out.feat, expected, atol=1e-6)
