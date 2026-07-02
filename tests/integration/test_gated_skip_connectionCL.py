# tests/integration/test_gated_skip_connectionCL.py

import sys
from pathlib import Path
import copy
import torch
import pytest
import torch.nn as nn
import mock_dependencies
from pointcept.models.utils.structure import Point
from pointcept.models.point_transformer_v3.new_modules import GatedSkipConnectionCL
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import SerializedUnpooling


@pytest.fixture(scope="module")
def cpu_device():
    return torch.device("cpu")


@pytest.fixture
def point_factory(cpu_device):
    def _make(
        num_child=8,
        num_parent=16,
        in_channels=4,
        skip_channels=6,
        seed=0,
    ):
        torch.manual_seed(seed)

        child = Point(
            feat=torch.randn(num_child, in_channels, device=cpu_device),
        )

        parent = Point(
            feat=torch.randn(num_parent, skip_channels, device=cpu_device),
        )

        inverse = torch.randint(
            low=0, high=num_child, size=(num_parent,), device=cpu_device
        )

        child["pooling_parent"] = parent
        child["pooling_inverse"] = inverse
        return child

    return _make


@pytest.fixture
def unpool_config(**overrides):
    kwargs = dict(
        in_channels=4,
        skip_channels=6,
        out_channels=8,
        norm_layer=None,
        act_layer=None,
        traceable=False,
    )
    kwargs.update(overrides)
    return kwargs


class TestGSCIntegration:
    def test_serialized_unpooling_constructs_gsc_when_enabled(self, cpu_device, unpool_config):
        layer = SerializedUnpooling(
            **unpool_config,
            enable_gsc=True,
        ).to(cpu_device)

        assert hasattr(layer, "gsc")
        assert isinstance(layer.gsc, GatedSkipConnectionCL)
        assert isinstance(layer.gsc.dec_proj, nn.Identity)
        assert layer.gsc.gate[0].in_features == 2 * unpool_config["out_channels"]
        assert layer.gsc.gate[-2].out_features == unpool_config["out_channels"]

    def test_serialized_unpooling_has_no_gsc_when_disabled(self, cpu_device, unpool_config):
        layer = SerializedUnpooling(
            **unpool_config,
            enable_gsc=False,
        ).to(cpu_device)

        assert not hasattr(layer, "gsc")

    def test_serialized_unpooling_forward_invokes_gsc_when_enabled(
        self, cpu_device, unpool_config, point_factory, monkeypatch
    ):
        layer = SerializedUnpooling(
            **unpool_config,
            enable_gsc=True,
        ).to(cpu_device)

        point = point_factory(
            num_child=8,
            num_parent=16,
            in_channels=unpool_config["in_channels"],
            skip_channels=unpool_config["skip_channels"],
            seed=1,
        )

        call_count = {"n": 0}
        original_forward = layer.gsc.forward

        def wrapped_forward(enc_feat, dec_feat):
            call_count["n"] += 1
            return original_forward(enc_feat, dec_feat)

        monkeypatch.setattr(layer.gsc, "forward", wrapped_forward)

        out = layer(point)

        assert call_count["n"] == 1, "Expected GSC to be invoked exactly once"
        assert hasattr(out, "feat")
        assert out.feat.shape == (16, unpool_config["out_channels"])
        assert torch.isfinite(out.feat).all()

    def test_serialized_unpooling_forward_with_gsc_has_grad_flow(
        self, cpu_device, unpool_config, point_factory
    ):
        layer = SerializedUnpooling(
            **unpool_config,
            enable_gsc=True,
        ).to(cpu_device)

        point = point_factory(
            num_child=8,
            num_parent=16,
            in_channels=unpool_config["in_channels"],
            skip_channels=unpool_config["skip_channels"],
            seed=2,
        )

        point.feat.requires_grad_(True)
        point["pooling_parent"].feat.requires_grad_(True)

        out = layer(point)
        loss = out.feat.mean()
        loss.backward()

        has_grad = False
        for p in layer.gsc.parameters():
            if p.requires_grad and p.grad is not None:
                assert torch.isfinite(p.grad).all()
                has_grad = True

        assert has_grad, "Expected gradient flow into GSC parameters"

    def test_serialized_unpooling_forward_without_gsc_uses_baseline_addition(
        self, cpu_device, unpool_config, point_factory
    ):
        layer = SerializedUnpooling(
            **unpool_config,
            enable_gsc=False,
        ).to(cpu_device)

        point = point_factory(
            num_child=8,
            num_parent=16,
            in_channels=unpool_config["in_channels"],
            skip_channels=unpool_config["skip_channels"],
            seed=3,
        )

        point_copy = copy.deepcopy(point)

        parent = point_copy["pooling_parent"]
        inverse = point_copy["pooling_inverse"]

        projected_child = layer.proj(point_copy)
        projected_parent = layer.proj_skip(parent)
        expected = projected_parent.feat + projected_child.feat[inverse]

        out = layer(point)

        assert torch.allclose(out.feat, expected, atol=1e-6), (
            "When enable_gsc=False, forward should reduce to classic skip addition"
        )

    def test_serialized_unpooling_gsc_path_differs_from_plain_addition(
        self, cpu_device, unpool_config, point_factory
    ):
        layer_gsc = SerializedUnpooling(
            **unpool_config,
            enable_gsc=True,
        ).to(cpu_device)

        layer_add = SerializedUnpooling(
            **unpool_config,
            enable_gsc=False,
        ).to(cpu_device)

        layer_add.proj.load_state_dict(layer_gsc.proj.state_dict())
        layer_add.proj_skip.load_state_dict(layer_gsc.proj_skip.state_dict())

        point_a = point_factory(
            num_child=8,
            num_parent=16,
            in_channels=unpool_config["in_channels"],
            skip_channels=unpool_config["skip_channels"],
            seed=4,
        )
        point_b = copy.deepcopy(point_a)

        out_gsc = layer_gsc(point_a)
        out_add = layer_add(point_b)

        assert out_gsc.feat.shape == out_add.feat.shape
        assert torch.isfinite(out_gsc.feat).all()
        assert torch.isfinite(out_add.feat).all()
        assert not torch.allclose(out_gsc.feat, out_add.feat), (
            "Expected gated fusion to differ from plain addition baseline"
        )

    def test_serialized_unpooling_traceable_keeps_unpooling_parent_reference(
        self, cpu_device, unpool_config, point_factory
    ):
        config = unpool_config.copy()
        config["traceable"] = True
        layer = SerializedUnpooling(
            **config,
            enable_gsc=True,
        ).to(cpu_device)

        point = point_factory(
            num_child=8,
            num_parent=16,
            in_channels=unpool_config["in_channels"],
            skip_channels=unpool_config["skip_channels"],
            seed=5,
        )

        # ذخیره رفرنس اصلی برای مقایسه در انتهای تست
        original_child_ref = point

        out = layer(point)

        # بررسی تکمیل‌شده:
        assert "unpooling_parent" in out, "Expected 'unpooling_parent' key when traceable is True"
        assert out["unpooling_parent"] is original_child_ref, "Expected the exact reference to the child point"
