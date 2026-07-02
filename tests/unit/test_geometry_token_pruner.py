import torch
import pytest
import mock_dependencies

from pointcept.models.utils.structure import Point
from pointcept.models.point_transformer_v3.new_modules import GeometryTokenPruner


@pytest.fixture
def cpu_device():
    return torch.device("cpu")


def make_point(
    batch_sizes=(16,),
    feat_dim=8,
    device=torch.device("cpu"),
    seed=0,
):
    torch.manual_seed(seed)

    num_points = sum(batch_sizes)

    feat = torch.randn(num_points, feat_dim, device=device)
    coord = torch.randn(num_points, 3, device=device)

    if len(batch_sizes) == 0:
        offset = torch.empty(0, dtype=torch.long, device=device)
    else:
        offset = torch.tensor(batch_sizes, dtype=torch.long, device=device).cumsum(0)

    serialized_order = [torch.arange(num_points, device=device)]
    serialized_inverse = [torch.arange(num_points, device=device)]

    return Point(
        feat=feat,
        coord=coord,
        offset=offset,
        serialized_order=serialized_order,
        serialized_inverse=serialized_inverse,
    )


def kept_counts_from_mask_and_offset(keep_mask, offset):
    counts = []
    start = 0
    for end in offset.tolist():
        counts.append(int(keep_mask[start:end].sum().item()))
        start = end
    return counts


def test_default_init():
    pruner = GeometryTokenPruner()

    assert pruner.prune_ratio == 0.25
    assert pruner.k == 8
    assert pruner.min_keep == 100
    assert pruner.score_weights == (1.0, 1.0, 0.5)


def test_custom_init():
    pruner = GeometryTokenPruner(
        prune_ratio=0.5,
        k=4,
        min_keep=8,
        score_weights=(0.5, 1.5, 2.0),
    )

    assert pruner.prune_ratio == 0.5
    assert pruner.k == 4
    assert pruner.min_keep == 8
    assert pruner.score_weights == (0.5, 1.5, 2.0)


def test_forward_returns_mask_and_offset_shapes(cpu_device):
    point = make_point(batch_sizes=(16,), feat_dim=8, device=cpu_device, seed=1)

    pruner = GeometryTokenPruner(
        prune_ratio=0.25,
        k=4,
        min_keep=4,
    ).to(cpu_device)

    keep_mask, new_offset = pruner(point)

    assert isinstance(keep_mask, torch.Tensor)
    assert isinstance(new_offset, torch.Tensor)

    assert keep_mask.shape == (16,)
    assert keep_mask.dtype == torch.bool
    assert keep_mask.device == cpu_device

    assert new_offset.shape == point.offset.shape
    assert new_offset.dtype == point.offset.dtype
    assert new_offset.device == cpu_device


def test_forward_outputs_are_valid(cpu_device):
    point = make_point(batch_sizes=(20,), feat_dim=8, device=cpu_device, seed=2)

    pruner = GeometryTokenPruner(
        prune_ratio=0.25,
        k=4,
        min_keep=4,
    ).to(cpu_device)

    keep_mask, new_offset = pruner(point)

    assert torch.isfinite(point.feat).all()
    assert torch.isfinite(point.coord).all()

    assert keep_mask.dtype == torch.bool
    assert keep_mask.numel() == point.feat.shape[0]

    assert new_offset.ndim == 1
    assert new_offset.numel() == point.offset.numel()
    assert torch.all(new_offset >= 0)
    assert torch.all(new_offset <= point.offset[-1])


def test_prune_ratio_reduces_or_keeps_valid_number_of_tokens(cpu_device):
    point = make_point(batch_sizes=(20,), feat_dim=8, device=cpu_device, seed=3)

    pruner = GeometryTokenPruner(
        prune_ratio=0.5,
        k=4,
        min_keep=4,
    ).to(cpu_device)

    keep_mask, new_offset = pruner(point)

    kept = int(keep_mask.sum().item())

    assert kept >= 4
    assert kept <= 20
    assert int(new_offset[-1].item()) == kept


def test_keeps_at_least_min_keep_per_batch(cpu_device):
    batch_sizes = (16, 20)
    min_keep = 6

    point = make_point(batch_sizes=batch_sizes, feat_dim=8, device=cpu_device, seed=4)

    pruner = GeometryTokenPruner(
        prune_ratio=0.75,
        k=4,
        min_keep=min_keep,
    ).to(cpu_device)

    keep_mask, new_offset = pruner(point)

    kept_counts = kept_counts_from_mask_and_offset(keep_mask, point.offset)

    for kept, batch_size in zip(kept_counts, batch_sizes):
        assert kept >= min(min_keep, batch_size)
        assert kept <= batch_size

    assert new_offset.tolist() == torch.tensor(kept_counts, device=cpu_device).cumsum(0).tolist()


def test_keeps_all_when_batch_size_is_below_or_equal_min_keep(cpu_device):
    batch_sizes = (3, 4)
    min_keep = 4

    point = make_point(batch_sizes=batch_sizes, feat_dim=8, device=cpu_device, seed=5)

    pruner = GeometryTokenPruner(
        prune_ratio=0.9,
        k=4,
        min_keep=min_keep,
    ).to(cpu_device)

    keep_mask, new_offset = pruner(point)

    assert keep_mask.all()
    assert int(keep_mask.sum().item()) == sum(batch_sizes)
    assert new_offset.tolist() == torch.tensor(batch_sizes, device=cpu_device).cumsum(0).tolist()


def test_multi_batch_offsets_match_kept_counts(cpu_device):
    batch_sizes = (12, 18, 24)

    point = make_point(batch_sizes=batch_sizes, feat_dim=8, device=cpu_device, seed=6)

    pruner = GeometryTokenPruner(
        prune_ratio=0.5,
        k=4,
        min_keep=4,
    ).to(cpu_device)

    keep_mask, new_offset = pruner(point)

    kept_counts = kept_counts_from_mask_and_offset(keep_mask, point.offset)
    expected_new_offset = torch.tensor(kept_counts, dtype=torch.long, device=cpu_device).cumsum(0)

    assert new_offset.tolist() == expected_new_offset.tolist()
    assert int(new_offset[-1].item()) == int(keep_mask.sum().item())


def test_empty_input_returns_empty_mask_and_zero_offset(cpu_device):
    point = make_point(batch_sizes=(), feat_dim=8, device=cpu_device, seed=7)

    pruner = GeometryTokenPruner(
        prune_ratio=0.25,
        k=4,
        min_keep=4,
    ).to(cpu_device)

    keep_mask, new_offset = pruner(point)

    assert keep_mask.shape == (0,)
    assert keep_mask.dtype == torch.bool
    assert keep_mask.device == cpu_device

    assert new_offset.shape == point.offset.shape
    assert new_offset.dtype == point.offset.dtype
    assert new_offset.device == cpu_device
    assert new_offset.numel() == 0


def test_k_larger_than_n_is_handled(cpu_device):
    point = make_point(batch_sizes=(5,), feat_dim=8, device=cpu_device, seed=8)

    pruner = GeometryTokenPruner(
        prune_ratio=0.5,
        k=32,
        min_keep=2,
    ).to(cpu_device)

    keep_mask, new_offset = pruner(point)

    assert keep_mask.shape == (5,)
    assert keep_mask.dtype == torch.bool
    assert int(keep_mask.sum().item()) >= 2
    assert int(keep_mask.sum().item()) <= 5
    assert int(new_offset[-1].item()) == int(keep_mask.sum().item())


def test_deterministic_for_same_input(cpu_device):
    point1 = make_point(batch_sizes=(20,), feat_dim=8, device=cpu_device, seed=9)
    point2 = make_point(batch_sizes=(20,), feat_dim=8, device=cpu_device, seed=9)

    pruner = GeometryTokenPruner(
        prune_ratio=0.25,
        k=4,
        min_keep=4,
    ).to(cpu_device)

    keep_mask1, new_offset1 = pruner(point1)
    keep_mask2, new_offset2 = pruner(point2)

    assert torch.equal(keep_mask1, keep_mask2)
    assert torch.equal(new_offset1, new_offset2)


def test_batch_awareness_offsets_do_not_mix_batches(cpu_device):
    batch_sizes = (10, 10)

    point = make_point(batch_sizes=batch_sizes, feat_dim=8, device=cpu_device, seed=10)

    pruner = GeometryTokenPruner(
        prune_ratio=0.5,
        k=4,
        min_keep=3,
    ).to(cpu_device)

    keep_mask, new_offset = pruner(point)

    kept_counts = kept_counts_from_mask_and_offset(keep_mask, point.offset)

    assert len(kept_counts) == len(batch_sizes)

    for kept, batch_size in zip(kept_counts, batch_sizes):
        assert kept >= 3
        assert kept <= batch_size

    assert new_offset.tolist() == torch.tensor(kept_counts, device=cpu_device).cumsum(0).tolist()


def test_kept_indices_can_index_feat_and_coord(cpu_device):
    point = make_point(batch_sizes=(16,), feat_dim=8, device=cpu_device, seed=11)

    pruner = GeometryTokenPruner(
        prune_ratio=0.25,
        k=4,
        min_keep=4,
    ).to(cpu_device)

    keep_mask, new_offset = pruner(point)

    kept_feat = point.feat[keep_mask]
    kept_coord = point.coord[keep_mask]

    assert kept_feat.shape[0] == int(keep_mask.sum().item())
    assert kept_coord.shape[0] == int(keep_mask.sum().item())
    assert kept_feat.shape[1] == point.feat.shape[1]
    assert kept_coord.shape[1] == 3
    assert int(new_offset[-1].item()) == kept_feat.shape[0]

