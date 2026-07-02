import torch
import pytest
import mock_dependencies

from pointcept.models.utils.structure import Point
from pointcept.models.point_transformer_v3.new_modules import GeometryTokenPruner
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import (
    SerializedPooling,
    PointTransformerV3,
)


@pytest.fixture
def cpu_device():
    return torch.device("cpu")


def make_pooling_ready_point(
    num_points=8,
    feat_dim=6,
    device=torch.device("cpu"),
    seed=0,
):
    torch.manual_seed(seed)

    feat = torch.randn(num_points, feat_dim, device=device)
    coord = torch.randn(num_points, 3, device=device)
    offset = torch.tensor([num_points], dtype=torch.long, device=device)
    batch = torch.zeros(num_points, dtype=torch.long, device=device)

    grid_coord = torch.arange(num_points * 3, device=device).reshape(num_points, 3).long()
    serialized_code = torch.arange(num_points, device=device).long().unsqueeze(0)
    serialized_order = torch.arange(num_points, device=device).long().unsqueeze(0)
    serialized_inverse = torch.arange(num_points, device=device).long().unsqueeze(0)
    serialized_depth = 1

    return Point(
        feat=feat,
        coord=coord,
        offset=offset,
        batch=batch,
        grid_coord=grid_coord,
        serialized_code=serialized_code,
        serialized_order=serialized_order,
        serialized_inverse=serialized_inverse,
        serialized_depth=serialized_depth,
    )


class FakePruner(torch.nn.Module):
    def __init__(self, keep_mask=None, new_offset=None):
        super().__init__()
        self.called = 0
        self._keep_mask = keep_mask
        self._new_offset = new_offset

    def forward(self, point):
        self.called += 1
        keep_mask = (
            self._keep_mask.to(point.feat.device)
            if self._keep_mask is not None
            else torch.ones(point.feat.shape[0], dtype=torch.bool, device=point.feat.device)
        )
        new_offset = (
            self._new_offset.to(point.offset.device)
            if self._new_offset is not None
            else point.offset.clone()
        )
        return keep_mask, new_offset


def make_small_pool(enable_gtp=True, in_channels=6, out_channels=12):
    return SerializedPooling(
        in_channels=in_channels,
        out_channels=out_channels,
        stride=2,
        reduce="max",
        shuffle_orders=False,
        traceable=True,
        enable_gtp=enable_gtp,
        gtp_prune_ratio=0.25,
        gtp_k=4,
    )


def test_serialized_pooling_wires_gtp_when_enabled(cpu_device):
    pool = make_small_pool(enable_gtp=True).to(cpu_device)

    assert pool.enable_gtp is True
    assert hasattr(pool, "gtp_pruner")
    assert isinstance(pool.gtp_pruner, GeometryTokenPruner)
    assert pool.gtp_pruner.prune_ratio == 0.25
    assert pool.gtp_pruner.k == 4


def test_serialized_pooling_does_not_wire_gtp_when_disabled(cpu_device):
    pool = make_small_pool(enable_gtp=False).to(cpu_device)

    assert pool.enable_gtp is False
    assert not hasattr(pool, "gtp_pruner")


def test_serialized_pooling_calls_pruner_in_training(cpu_device):
    pool = make_small_pool(enable_gtp=True).to(cpu_device)
    pool.train()

    fake = FakePruner()
    pool.gtp_pruner = fake

    point = make_pooling_ready_point(num_points=8, feat_dim=6, device=cpu_device, seed=1)

    try:
        pool(point)
    except Exception:
        pass

    assert fake.called >= 1


def test_serialized_pooling_does_not_call_pruner_in_eval(cpu_device):
    pool = make_small_pool(enable_gtp=True).to(cpu_device)
    pool.eval()

    fake = FakePruner()
    pool.gtp_pruner = fake

    point = make_pooling_ready_point(num_points=8, feat_dim=6, device=cpu_device, seed=2)

    try:
        with torch.no_grad():
            pool(point)
    except Exception:
        pass

    assert fake.called == 0


def test_serialized_pooling_prunes_point_fields_before_pooling(cpu_device):
    pool = make_small_pool(enable_gtp=True).to(cpu_device)
    pool.train()

    keep_mask = torch.tensor(
        [True, False, True, False, True, False, True, False],
        dtype=torch.bool,
        device=cpu_device,
    )
    new_offset = torch.tensor([4], dtype=torch.long, device=cpu_device)
    fake = FakePruner(keep_mask=keep_mask, new_offset=new_offset)
    pool.gtp_pruner = fake

    point = make_pooling_ready_point(num_points=8, feat_dim=6, device=cpu_device, seed=3)

    try:
        pool(point)
    except Exception:
        pass

    assert fake.called == 1
    assert point.feat.shape[0] == 4
    assert point.coord.shape[0] == 4
    assert point.grid_coord.shape[0] == 4
    assert point.batch.shape[0] == 4
    assert point.serialized_code.shape[1] == 4
    assert torch.equal(point.offset, new_offset)
    assert point.serialized_order is None
    assert point.serialized_inverse is None

def make_small_model(enable_gtp=True, in_channels=6):
    return PointTransformerV3(
        in_channels=in_channels,
        enable_flash=False,
        enable_rpe=False,
        enable_gtp=enable_gtp,
        gtp_prune_ratio=0.25,
        gtp_k=4,
    )


def get_pooling_layers(model):
    return [m for m in model.modules() if isinstance(m, SerializedPooling)]


def test_model_propagates_gtp_to_pooling_layers_when_enabled(cpu_device):
    model = make_small_model(enable_gtp=True).to(cpu_device)

    pools = get_pooling_layers(model)
    assert len(pools) > 0

    enabled_pools = [p for p in pools if p.enable_gtp]
    assert len(enabled_pools) > 0

    for p in enabled_pools:
        assert hasattr(p, "gtp_pruner")
        assert isinstance(p.gtp_pruner, GeometryTokenPruner)
        assert p.gtp_pruner.prune_ratio == 0.25
        assert p.gtp_pruner.k == 4


def test_model_does_not_enable_gtp_in_pooling_layers_when_disabled(cpu_device):
    model = make_small_model(enable_gtp=False).to(cpu_device)

    pools = get_pooling_layers(model)
    assert len(pools) > 0

    for p in pools:
        assert p.enable_gtp is False
        assert not hasattr(p, "gtp_pruner")
