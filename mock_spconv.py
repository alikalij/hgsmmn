"""Mock spconv برای تست CPU-only"""
import torch
import torch.nn as nn

class SparseConvTensor:
    def __init__(self, features, indices, spatial_shape, batch_size):
        self.features = features
        self.indices = indices
        self.spatial_shape = spatial_shape
        self.batch_size = batch_size
    
    def dense(self):
        return self.features

class SubMConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, 1, bias=bias)
    
    def forward(self, x):
        x.features = self.conv(x.features.unsqueeze(-1)).squeeze(-1)
        return x

class SparseConv3d(SubMConv3d):
    pass

class SparseInverseConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, 1, bias=bias)
    
    def forward(self, x):
        x.features = self.conv(x.features.unsqueeze(-1)).squeeze(-1)
        return x

class SparseSequential(nn.Sequential):
    def forward(self, x):
        for module in self:
            x = module(x)
        return x
