import sys
import types
import importlib

import pytest
import torch


def _install_torch_scatter_polyfill():
    if "torch_scatter" in sys.modules:
        return

    torch_scatter = types.ModuleType("torch_scatter")

    def segment_csr(src, indptr, reduce="sum"):
        outputs = []

        for start, end in zip(indptr[:-1].tolist(), indptr[1:].tolist()):
            segment = src[start:end]

            if segment.numel() == 0:
                outputs.append(src.new_zeros(src.shape[1:]))
            elif reduce in ("sum", "add"):
                outputs.append(segment.sum(dim=0))
            elif reduce == "mean":
                outputs.append(segment.mean(dim=0))
            elif reduce == "min":
                outputs.append(segment.min(dim=0).values)
            elif reduce == "max":
                outputs.append(segment.max(dim=0).values)
            else:
                raise NotImplementedError(f"Unsupported reduce mode: {reduce}")

        if not outputs:
            return src.new_empty((0, *src.shape[1:]))

        return torch.stack(outputs, dim=0)

    torch_scatter.segment_csr = segment_csr
    sys.modules["torch_scatter"] = torch_scatter


_install_torch_scatter_polyfill()

try:
    import mock_dependencies  # noqa: F401
except ImportError:
    pass


def _import_point_transformer_v3():
    candidates = [
        "pointcept.models.point_transformer_v3.point_transformer_v3m1_base",
        "point_transformer_v3m1_base",
    ]

    last_error = None
    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
            return module.PointTransformerV3
        except ImportError as exc:
            last_error = exc

    raise last_error


PointTransformerV3 = _import_point_transformer_v3()


@pytest.fixture(autouse=True)
def cpu_determinism():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    yield


def _make_serialized_fields(grid_coord, offset, num_orders=4):
    device = grid_coord.device
    n_points = grid_coord.shape[0]

    batch = torch.empty(n_points, dtype=torch.long, device=device)
    start = 0
    for batch_idx, end in enumerate(offset.tolist()):
        batch[start:end] = batch_idx
        start = end

    x = grid_coord[:, 0].long()
    y = grid_coord[:, 1].long()
    z = grid_coord[:, 2].long()

    keys = [
        batch * 1_000_000 + x * 10_000 + y * 100 + z,
        batch * 1_000_000 + z * 10_000 + y * 100 + x,
        batch * 1_000_000 + y * 10_000 + x * 100 + z,
        batch * 1_000_000 + x * 10_000 + z * 100 + y,
    ]

    orders = []
    inverses = []
    codes = []

    for idx in range(num_orders):
        code = keys[idx % len(keys)]
        order = torch.argsort(code, stable=True)

        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(n_points, device=device)

        orders.append(order)
        inverses.append(inverse)
        codes.append(code)

    return {
        "serialized_order": torch.stack(orders, dim=0),
        "serialized_inverse": torch.stack(inverses, dim=0),
        "serialized_code": torch.stack(codes, dim=0),
        "serialized_depth": torch.tensor(4, dtype=torch.long, device=device),
    }


def _make_point_data(batch_sizes=(48,), in_channels=6):
    coords = []
    feats = []
    offsets = []

    cumulative = 0
    for batch_idx, count in enumerate(batch_sizes):
        side = int(torch.ceil(torch.tensor(float(count)).pow(1 / 3)).item())

        local_grid = torch.stack(
            torch.meshgrid(
                torch.arange(side),
                torch.arange(side),
                torch.arange(side),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(-1, 3)[:count]

        local_grid = local_grid + torch.tensor([batch_idx * 16, 0, 0])

        coords.append(local_grid.float())
        feats.append(torch.randn(count, in_channels))

        cumulative += count
        offsets.append(cumulative)

    coord = torch.cat(coords, dim=0)
    grid_coord = coord.long()
    feat = torch.cat(feats, dim=0)
    offset = torch.tensor(offsets, dtype=torch.long)

    data_dict = {
        "coord": coord,
        "grid_coord": grid_coord,
        "feat": feat,
        "offset": offset,
    }
    data_dict.update(_make_serialized_fields(grid_coord, offset, num_orders=4))
    return data_dict


@pytest.fixture
def ptv3_cpu_model():
    model = PointTransformerV3(
        in_channels=6,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2),
        enc_depths=(1, 1, 1),
        enc_channels=(16, 32, 64),
        enc_num_head=(2, 4, 8),
        enc_patch_size=(16, 16, 16),
        dec_depths=(1, 1),
        dec_channels=(16, 32),
        dec_num_head=(2, 4),
        dec_patch_size=(16, 16),
        mlp_ratio=2,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        pre_norm=True,
        shuffle_orders=False,
        enable_rpe=False,
        enable_flash=False,
        upcast_attention=True,
        upcast_softmax=True,
        cls_mode=False,
        enable_spe=True,
        enable_gsc=True,
        enable_grab=True,
        grab_hidden_dim=16,
        grab_use_distance=True,
        grab_per_head=True,
        grab_init_scale=0.01,
        enable_gtp=True,
        gtp_prune_ratio=[0.1, 0.2],
        gtp_k=4,
        enable_gct=True,
        gct_num_anchors=2,
        spe_dim=16,
    )
    return model.cpu()


def test_ptv3_backbone_forward_cpu_all_new_modules_enabled(ptv3_cpu_model):
    model = ptv3_cpu_model.eval()
    data_dict = _make_point_data(batch_sizes=(48,), in_channels=6)

    with torch.no_grad():
        output = model(data_dict)

    # تغییر: بررسی ویژگی feat داخل شیء Point
    assert hasattr(output, "feat")
    assert isinstance(output.feat, torch.Tensor)
    assert output.feat.ndim == 2
    assert output.feat.shape[0] > 0
    assert output.feat.shape[0] <= data_dict["feat"].shape[0]
    assert output.feat.shape[1] == 16
    assert torch.isfinite(output.feat).all()


def test_ptv3_backbone_backward_cpu_all_new_modules_enabled(ptv3_cpu_model):
    model = ptv3_cpu_model.eval()
    data_dict = _make_point_data(batch_sizes=(256,), in_channels=6)

    output = model(data_dict)
    loss = output.feat.pow(2).mean()

    assert torch.isfinite(loss)

    loss.backward()

    finite_grad_count = 0
    nonzero_grad_count = 0

    for param in model.parameters():
        if param.grad is None:
            continue

        assert torch.isfinite(param.grad).all()
        finite_grad_count += 1

        if param.grad.abs().sum().item() > 0:
            nonzero_grad_count += 1

    assert finite_grad_count > 0
    assert nonzero_grad_count > 0


def test_ptv3_backbone_eval_output_is_deterministic(ptv3_cpu_model):
    model = ptv3_cpu_model.eval()
    data_dict = _make_point_data(batch_sizes=(40,), in_channels=6)

    with torch.no_grad():
        torch.manual_seed(123)
        output_1 = model(data_dict)

        torch.manual_seed(123)
        output_2 = model(data_dict)

    assert output_1.feat.shape == output_2.feat.shape
    assert torch.allclose(output_1.feat, output_2.feat, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("batch_sizes", [(32,), (64, 80)])
def test_ptv3_backbone_forward_cpu_with_different_offsets(ptv3_cpu_model, batch_sizes):
    model = ptv3_cpu_model.eval()
    data_dict = _make_point_data(batch_sizes=batch_sizes, in_channels=6)

    with torch.no_grad():
        output = model(data_dict)

    assert hasattr(output, "feat")
    assert isinstance(output.feat, torch.Tensor)
    assert output.feat.ndim == 2
    assert output.feat.shape[0] > 0
    assert output.feat.shape[0] <= sum(batch_sizes)
    assert output.feat.shape[1] == 16
    assert torch.isfinite(output.feat).all()


def test_ptv3_backbone_new_modules_are_activated(ptv3_cpu_model):
    model = ptv3_cpu_model.eval()
    data_dict = _make_point_data(batch_sizes=(48,), in_channels=6)

    activations = {}

    def make_hook(name):
        def hook(module, inputs, output):
            activations[name] = True

        return hook

    hooks = []
    target_keywords = (
        "spe",
        "gsc",
        "grab",
        "gtp",
        "gct",
        "structural",
        "globalcontext",
        "geometry",
    )

    for name, module in model.named_modules():
        module_name = name.lower()
        class_name = module.__class__.__name__.lower()

        if any(keyword in module_name or keyword in class_name for keyword in target_keywords):
            hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        model(data_dict)

    for hook in hooks:
        hook.remove()

    assert len(hooks) > 0, "No candidate new modules were found for activation hooks."
    assert len(activations) > 0, "No new module activation was observed during forward."
