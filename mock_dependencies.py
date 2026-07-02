# mock_dependencies.py
"""
Mock کردن وابستگی‌های CUDA-only برای اجرای Point Transformer V3 روی CPU
"""

import sys
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib.machinery
from importlib.machinery import ModuleSpec

# ============================================================================
# Mock spconv
# ============================================================================
mock_spconv = types.ModuleType('spconv')
mock_spconv_pytorch = types.ModuleType('spconv.pytorch')
mock_spconv_modules = types.ModuleType('spconv.pytorch.modules')

class SparseConvTensor:
    """Mock sparse convolution tensor"""
    def __init__(self, features, indices, spatial_shape, batch_size):
        self.features = features
        self.indices = indices
        self.spatial_shape = spatial_shape
        self.batch_size = batch_size
    
    def dense(self):
        return self.features

    def replace_feature(self, new_features):
        return SparseConvTensor(new_features, self.indices, self.spatial_shape, self.batch_size)

class SparseModule(nn.Module):
    """Base class for sparse modules"""
    def __init__(self):
        super().__init__()

class SubMConv3d(SparseModule):
    """Mock submanifold sparse 3D convolution"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 dilation=1, groups=1, bias=True, indice_key=None, algo=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.indice_key = indice_key
        self.algo = algo
        
        # استفاده از Linear برای شبیه‌سازی convolution
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, input):
        """
        ورودی می‌تونه:
        1. SparseConvTensor باشه
        2. یک Point object با attribute feat باشه
        3. یک dict با key 'feat' باشه
        """
        # تشخیص نوع ورودی
        if isinstance(input, SparseConvTensor):
            # حالت 1: SparseConvTensor
            features = input.features
            output_features = self.linear(features)
            return input.replace_feature(output_features)
        
        elif hasattr(input, 'feat'):
            # حالت 2: Point object
            features = input.feat
            output_features = self.linear(features)
            input.feat = output_features
            return input
        
        elif isinstance(input, dict) and 'feat' in input:
            # حالت 3: dictionary
            features = input['feat']
            output_features = self.linear(features)
            input['feat'] = output_features
            return input
        
        else:
            # حالت 4: tensor معمولی (fallback)
            return self.linear(input)

class SparseConv3d(SubMConv3d):
    """Mock sparse 3D convolution"""
    pass

class SparseInverseConv3d(SubMConv3d):
    """Mock inverse sparse 3D convolution"""
    pass

class SparseSequential(nn.Sequential):
    """Sequential container for sparse modules"""
    def forward(self, input):
        for module in self:
            input = module(input)
        return input

class SparseMaxPool3d(SparseModule):
    """Mock sparse max pooling"""
    def __init__(self, kernel_size, stride=1, padding=0, dilation=1, indice_key=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.indice_key = indice_key

    def forward(self, input):
        # Mock: فقط ورودی را برمی‌گرداند
        return input

class ToDense(SparseModule):
    """Convert sparse tensor to dense"""
    def forward(self, input):
        if isinstance(input, SparseConvTensor):
            return input.dense()
        elif hasattr(input, 'feat'):
            return input.feat
        elif isinstance(input, dict) and 'feat' in input:
            return input['feat']
        return input

# تنظیم ماژول‌های spconv
mock_spconv_pytorch.SparseConvTensor = SparseConvTensor
mock_spconv_pytorch.SparseModule = SparseModule
mock_spconv_pytorch.SubMConv3d = SubMConv3d
mock_spconv_pytorch.SparseConv3d = SparseConv3d
mock_spconv_pytorch.SparseInverseConv3d = SparseInverseConv3d
mock_spconv_pytorch.SparseSequential = SparseSequential
mock_spconv_pytorch.SparseMaxPool3d = SparseMaxPool3d
mock_spconv_pytorch.ToDense = ToDense

# ماژول modules
mock_modules = types.ModuleType('spconv.pytorch.modules')
mock_modules.SparseModule = SparseModule
def is_spconv_module(module):
    return isinstance(module, SparseModule)
mock_modules.is_spconv_module = is_spconv_module
mock_spconv_pytorch.modules = mock_modules

sys.modules['spconv'] = mock_spconv
sys.modules['spconv.pytorch'] = mock_spconv_pytorch
sys.modules['spconv.pytorch.modules'] = mock_modules
print("✓ Mock شد: spconv")

# ============================================================================
# Mock pointops
# ============================================================================
mock_pointops = types.ModuleType('pointops')

def knn_query(k, xyz, new_xyz, offset, new_offset):
    """Mock KNN query"""
    num_new_points = new_xyz.shape[0]
    indices = torch.randint(0, xyz.shape[0], (num_new_points, k), dtype=torch.long)
    distances = torch.rand(num_new_points, k)
    return indices, distances

def grouping(features, indices):
    """Mock grouping"""
    num_groups, k = indices.shape
    C = features.shape[1]
    grouped_features = torch.zeros(num_groups, k, C)
    for i in range(num_groups):
        grouped_features[i, :, :] = features[indices[i].long(), :]
    return grouped_features

mock_pointops.knn_query = knn_query
mock_pointops.knn_query_and_group = lambda k, xyz, new_xyz, offset, new_offset, feat: (
    knn_query(k, xyz, new_xyz, offset, new_offset)[0], 
    grouping(feat, knn_query(k, xyz, new_xyz, offset, new_offset)[0])
)
mock_pointops.grouping = grouping

sys.modules['pointops'] = mock_pointops
print("✓ Mock شد: pointops")

# ============================================================================
# Mock flash_attn
# ============================================================================
mock_flash_attn = types.ModuleType('flash_attn')
mock_flash_attn_modules = types.ModuleType('flash_attn.modules')
mock_flash_attn_modules_mha = types.ModuleType('flash_attn.modules.mha')

class FlashSelfAttention(nn.Module):
    """Mock flash attention"""
    def __init__(self, embed_dim, num_heads, qkv_bias=False, attn_drop=0., proj_drop=0., causal=False, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.causal = causal
        
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn_scores += mask
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_drop(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v).permute(0, 2, 1, 3).reshape(B, N, C)
        output = self.proj(attn_output)
        output = self.proj_drop(output)
        return output

mock_flash_attn_modules_mha.FlashSelfAttention = FlashSelfAttention
mock_flash_attn.modules = mock_flash_attn_modules
mock_flash_attn.modules.mha = mock_flash_attn_modules_mha

sys.modules['flash_attn'] = mock_flash_attn
sys.modules['flash_attn.modules'] = mock_flash_attn_modules
sys.modules['flash_attn.modules.mha'] = mock_flash_attn_modules_mha
print("✓ Mock شد: flash_attn")
# ============================================================================
# Mock peft
# ============================================================================
mock_peft = types.ModuleType('peft')
mock_peft.__spec__ = importlib.machinery.ModuleSpec('peft', None)
mock_peft.LoraConfig = type('LoraConfig', (), {'__init__': lambda self, *args, **kwargs: None})
mock_peft.get_peft_model = lambda model, config: model
sys.modules['peft'] = mock_peft
print("✓ Mock شد: peft")
# ============================================================================
# Mock addict (attribute-style dict access)
# ============================================================================
mock_addict = types.ModuleType('addict')

class MockDict(dict):
    """Mock implementation of addict.Dict - dict with attribute access"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict) and not isinstance(value, MockDict):
                self[key] = MockDict(value)
    
    def __getattr__(self, key):
        try:
            value = self[key]
            if isinstance(value, dict) and not isinstance(value, MockDict):
                value = MockDict(value)
                self[key] = value
            return value
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")
    
    def __setattr__(self, key, value):
        self[key] = value
    
    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")

mock_addict.Dict = MockDict
sys.modules['addict'] = mock_addict
print("✓ Mock شد: addict")

# ============================================================================
# Mock torch_cluster
# ============================================================================
mock_cluster = types.ModuleType('torch_cluster')

class MockCluster:
    @staticmethod
    def knn(k, xyz, new_xyz, offset, new_offset):
        num_new_points = new_xyz.shape[0]
        indices = torch.randint(0, xyz.shape[0], (num_new_points, k), dtype=torch.long)
        return indices
    
    @staticmethod
    def radius(radius, xyz, new_xyz, offset, new_offset):
        num_new_points = new_xyz.shape[0]
        indices = torch.randint(0, xyz.shape[0], (num_new_points, 10), dtype=torch.long)
        return indices

    @staticmethod
    def fps(ratio, xyz, offset):
        num_points = xyz.shape[0]
        num_samples = int(ratio * num_points)
        indices = torch.randperm(num_points)[:num_samples]
        return indices
        
mock_cluster.knn = MockCluster.knn
mock_cluster.radius = MockCluster.radius
mock_cluster.fps = MockCluster.fps

sys.modules['torch_cluster'] = mock_cluster
print("✓ Mock شد: torch_cluster")
# ============================================================================
# Mock torch_scatter
# ============================================================================
mock_scatter = types.ModuleType('torch_scatter')

def segment_csr(src, ptr, reduce="mean"):
    if not isinstance(src, torch.Tensor):
        raise TypeError(f"segment_csr expected Tensor src, got {type(src)}")
        
    num_segments = ptr.size(0) - 1
    channels = src.size(1) if src.dim() > 1 else 1
    
    # ساخت اندیس‌ها از روی اشاره‌گرها (ptr) با عملیات برداری
    index = torch.zeros(src.size(0), dtype=torch.long, device=src.device)
    for i in range(num_segments):
        if ptr[i] < ptr[i+1]:
            index[ptr[i]:ptr[i+1]] = i
            
    if src.dim() > 1:
        index = index.unsqueeze(1).expand(-1, channels)
        
    out = torch.zeros((num_segments, channels) if src.dim() > 1 else num_segments, 
                      dtype=src.dtype, device=src.device)
    
    # استفاده از scatter_reduce_ بومی PyTorch برای بالاترین سرعت
    if reduce == "mean":
        out.scatter_reduce_(0, index, src, reduce="mean", include_self=False)
    elif reduce == "max":
        out.scatter_reduce_(0, index, src, reduce="amax", include_self=False)
    elif reduce in ("sum", "add"):
        out.scatter_reduce_(0, index, src, reduce="sum", include_self=False)
        
    return out

def scatter_max(src, index, dim_size=None, dim=-1):
    if dim_size is None:
        dim_size = int(index.max()) + 1
    out = torch.zeros((dim_size, src.size(1)), dtype=src.dtype, device=src.device)
    idx_exp = index.unsqueeze(-1).expand_as(src)
    out.scatter_reduce_(0, idx_exp, src, reduce="amax", include_self=False)
    # برگرداندن تنسور خالی برای آرگومان دوم (اندیس‌ها) جهت جلوگیری از خطای Unpacking
    return out, torch.zeros_like(out, dtype=torch.long)

def scatter_mean(src, index, dim_size=None, dim=-1):
    if dim_size is None:
        dim_size = int(index.max()) + 1
    out = torch.zeros((dim_size, src.size(1)), dtype=src.dtype, device=src.device)
    idx_exp = index.unsqueeze(-1).expand_as(src)
    out.scatter_reduce_(0, idx_exp, src, reduce="mean", include_self=False)
    return out

def scatter_softmax(src, index, dim=-1, dim_size=None):
    """
    Mock implementation for scatter_softmax.
    جهت عبور از خطاهای ایمپورت و اجرای تست در محیط CPU
    """
    # یک شبیه‌سازی ساده که فقط ابعاد تنسور را حفظ کند و باعث خطا نشود
    return torch.nn.functional.softmax(src, dim=dim)

mock_scatter.segment_csr = segment_csr
mock_scatter.scatter_max = scatter_max
mock_scatter.scatter_mean = scatter_mean
# اصلاح scatter_sum برای هماهنگی با index به جای ptr
mock_scatter.scatter_sum = lambda src, index, dim=-1, dim_size=None, **kwargs: torch.zeros_like(src) 
# اضافه کردن scatter_softmax
mock_scatter.scatter_softmax = scatter_softmax

# جایگذاری بی‌قید و شرط ماک بهینه در سیستم
sys.modules['torch_scatter'] = mock_scatter

#sys.modules['torch_scatter'] = _mock_torch_scatter
print("✓ Mock شد: torch_scatter (شامل segment_csr, scatter_max, scatter_mean, scatter_softmax)")
# ============================================================================
# Mock wandb (Weights & Biases tracking)
# ============================================================================
mock_wandb = types.ModuleType('wandb')

class MockWandbRun:
    def __init__(self):
        self.id = 'mock_run_id'
        self.name = 'mock_run'
        self.dir = './wandb'
    
    def log(self, *args, **kwargs):
        pass
    
    def finish(self):
        pass

mock_wandb.init = lambda *args, **kwargs: MockWandbRun()
mock_wandb.log = lambda *args, **kwargs: None
mock_wandb.finish = lambda *args, **kwargs: None
mock_wandb.run = None

sys.modules['wandb'] = mock_wandb
print("✓ Mock شد: wandb")
# ============================================================================
# Mock timm
# ============================================================================
# 1. ساخت ماژول اصلی timm
mock_timm = types.ModuleType('timm')
mock_timm.__spec__ = ModuleSpec(name="timm", loader=None)

# 2. تابع مجازی trunc_normal_
def mock_trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return tensor
mock_timm.trunc_normal_ = mock_trunc_normal_

# 3. ساخت زیرماژول timm.layers
mock_timm_layers = types.ModuleType('timm.layers')
mock_timm_layers.__spec__ = ModuleSpec(name="timm.layers", loader=None)

# 4. اضافه کردن DropPath و trunc_normal_ به timm.layers
class MockDropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x):
        return x
        
mock_timm_layers.DropPath = MockDropPath
mock_timm_layers.trunc_normal_ = mock_trunc_normal_ # <-- اضافه شدن این خط

# 5. ثبت هر دو در sys.modules
sys.modules['timm'] = mock_timm
sys.modules['timm.layers'] = mock_timm_layers
print("✓ Mock شد: timm و timm.layers (شامل DropPath و trunc_normal_)")
# ============================================================================
# Mock einops
# ============================================================================
mock_einops = types.ModuleType('einops')
mock_einops.__spec__ = ModuleSpec(name="einops", loader=None)

def mock_rearrange(tensor, pattern, **kwargs):
    return tensor

def mock_repeat(tensor, pattern, **kwargs):
    return tensor
    
def mock_reduce(tensor, pattern, reduction, **kwargs):
    return tensor

mock_einops.rearrange = mock_rearrange
mock_einops.repeat = mock_repeat
mock_einops.reduce = mock_reduce

sys.modules['einops'] = mock_einops
print("✓ Mock شد: einops")
# ============================================================================
# Mock torch_geometric
# ============================================================================
mock_tg = types.ModuleType('torch_geometric')
mock_tg.__spec__ = ModuleSpec(name='torch_geometric', loader=None)

mock_tg_utils = types.ModuleType('torch_geometric.utils')
mock_tg_utils.__spec__ = ModuleSpec(name='torch_geometric.utils', loader=None)

def mock_scatter(src, index=None, dim=0, dim_size=None, reduce='sum'):
    return src

mock_tg_utils.scatter = mock_scatter

mock_tg_nn = types.ModuleType('torch_geometric.nn')
mock_tg_nn.__spec__ = ModuleSpec(name='torch_geometric.nn', loader=None)

mock_tg_nn_pool = types.ModuleType('torch_geometric.nn.pool')
mock_tg_nn_pool.__spec__ = ModuleSpec(name='torch_geometric.nn.pool', loader=None)

def mock_voxel_grid(pos, size, batch=None, start=None, end=None):
    if hasattr(pos, 'shape'):
        return torch.arange(pos.shape[0], dtype=torch.long, device=pos.device)
    return torch.tensor([], dtype=torch.long)

mock_tg_nn_pool.voxel_grid = mock_voxel_grid

mock_tg.nn = mock_tg_nn
mock_tg.utils = mock_tg_utils
mock_tg_nn.pool = mock_tg_nn_pool

sys.modules['torch_geometric'] = mock_tg
sys.modules['torch_geometric.utils'] = mock_tg_utils
sys.modules['torch_geometric.nn'] = mock_tg_nn
sys.modules['torch_geometric.nn.pool'] = mock_tg_nn_pool

print("✓ Mock شد: torch_geometric")
# ============================================================================
# Mock torch-sparse & torch-spline-conv (To prevent WinError 127 / Crash)
# ============================================================================
for mod in ["torch_sparse", "torch_spline_conv", "torch_geometric.nn.conv.utils"]:
    if mod not in sys.modules:
        sys.modules[mod] = types.ModuleType(mod)
        print(f"✓ Mock شد: {mod} (جلوگیری از کرش DLL)")