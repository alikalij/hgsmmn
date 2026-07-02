import torch
import torch.nn as nn
from pointcept.models.utils.structure import Point
import torch.nn.functional as F
from torch_scatter import scatter_max, scatter_mean
from pointcept.models.modules import PointModule, PointSequential


class GeometryAwareRelativeAttentionBias(nn.Module):
    """
    Geometry-aware Relative Attention Bias (GRAB)
    
    Learns continuous attention bias from 3D geometric relationships.
    Compatible with both flash and non-flash attention paths.
    
    Args:
        num_heads: Number of attention heads
        hidden_dim: Hidden dimension of MLP (default: 32 for memory efficiency)
        use_distance: Whether to include Euclidean distance as 4th feature
        per_head: Whether to generate per-head bias (True) or shared bias (False)
        init_scale: Initial scale for learnable alpha parameter
    """
    def __init__(
        self,
        num_heads,
        hidden_dim=32,
        use_distance=True,
        per_head=True,
        init_scale=0.01,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.use_distance = use_distance
        self.per_head = per_head
        
        # Input: [dx, dy, dz] or [dx, dy, dz, dist]
        input_dim = 4 if use_distance else 3
        output_dim = num_heads if per_head else 1
        
        # Three-layer MLP for geometric encoding
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )
        
        # Learnable scale (starts small for training stability)
        self.alpha = nn.Parameter(torch.tensor(init_scale))
        
        # Initialize with small weights
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, rel_pos):
        """
        Args:
            rel_pos: (N', K, K, 3) relative positions from get_rel_pos()
        
        Returns:
            bias: (N', H, K, K) attention bias
        """
        # rel_pos shape: (N', K, K, 3)
        geo_features = rel_pos.float()
        
        if self.use_distance:
            # Add Euclidean distance as 4th feature
            dist = torch.norm(geo_features, dim=-1, keepdim=True)  # (N', K, K, 1)
            # Normalize to [0, 1] for stability
            dist = dist / (dist.max() + 1e-6)
            geo_features = torch.cat([geo_features, dist], dim=-1)  # (N', K, K, 4)
        
        # MLP: (N', K, K, input_dim) -> (N', K, K, H) or (N', K, K, 1)
        bias = self.mlp(geo_features)
        
        # Reshape to (N', H, K, K)
        if self.per_head:
            bias = bias.permute(0, 3, 1, 2).contiguous()  # (N', H, K, K)
        else:
            bias = bias.permute(0, 3, 1, 2)  # (N', 1, K, K)
            bias = bias.expand(-1, self.num_heads, -1, -1)  # (N', H, K, K)
        
        # Apply learnable scale
        return self.alpha * bias


class GeometryTokenPruner(nn.Module):
    def __init__(self, prune_ratio=0.25, k=8, min_keep=100, score_weights=(1.0, 1.0, 0.5)):
        super().__init__()
        self.prune_ratio = prune_ratio
        self.k = k  # Window size for 1D local neighborhood
        self.min_keep = min_keep
        self.score_weights = tuple(score_weights)
        self.w_feat, self.w_geom, self.w_bound = score_weights

    def forward(self, point: Point):
        feat = point.feat
        coord = point.coord
        offset = point.offset
        
        # 1. استفاده از اولین آرایه مرتب‌شده (z-order curve) برای یافتن همسایگی محلی بدون OOM
        # order shape: [num_orders, N], ما از order اول استفاده می کنیم
        order = point.serialized_order[0] 
        inverse = point.serialized_inverse[0]
        
        # مرتب‌سازی ویژگی‌ها و مختصات بر اساس order
        feat_sorted = feat[order]
        coord_sorted = coord[order]
        
        N = feat.shape[0]
        device = feat.device
        
        # guard
        if N == 0:
            keep_masks = torch.zeros(0, dtype=torch.bool, device=device)
            new_offset = torch.zeros_like(offset)
            return keep_masks, new_offset

        # make unfold output length exactly N
        window_k = min(self.k, N)
        # 2. ایجاد پنجره محلی 1D (جایگزین فوق‌سریع و کم‌مصرف برای KNN)
        pad_left = window_k  // 2
        pad_right = window_k  - 1 - pad_left
        
        # پدینگ برای جلوگیری از خطای مرزها
        feat_padded = F.pad(
            feat_sorted.unsqueeze(0).transpose(1, 2), 
            (pad_left, pad_right), 
            mode='replicate'
        ).transpose(1, 2).squeeze(0)

        coord_padded = F.pad(
            coord_sorted.unsqueeze(0).transpose(1, 2), 
            (pad_left, pad_right), 
            mode='replicate'
        ).transpose(1, 2).squeeze(0)

        # استخراج پنجره‌ها با سایز K
        feat_windows = feat_padded.unfold(0, window_k, 1) # [N, C, K]
        coord_windows = coord_padded.unfold(0, window_k, 1) # [N, 3, K]
        
        # 3. محاسبه امتیازات روی آرایه مرتب‌شده
        feat_mean = feat_windows.mean(dim=2)
        feat_contrast_sorted = torch.norm(feat_sorted - feat_mean, dim=1)
        
        geom_var_sorted = coord_windows.var(dim=2, unbiased=False).sum(dim=1)

        # restore to original order
        feat_contrast = feat_contrast_sorted[inverse]
        geom_var = geom_var_sorted[inverse]

        keep_masks = torch.zeros(N, dtype=torch.bool, device=device)
        new_offset = torch.zeros_like(offset)
        
        # 4. پردازش نمونه به نمونه (Batch-aware) با استفاده از offset
        start_idx = 0
        for i in range(len(offset)):
            end_idx = offset[i].item()
            b_size = end_idx - start_idx

            # اگر نقاط کمتر از حداقل مجاز بود، همه را نگه دار
            if b_size <= self.min_keep:
                keep_masks[start_idx:end_idx] = True
                num_kept = b_size
            else:
                b_feat_c = feat_contrast[start_idx:end_idx]
                b_geom_v = geom_var[start_idx:end_idx]

                b_feat_c = (b_feat_c - b_feat_c.min()) / (b_feat_c.max() - b_feat_c.min() + 1e-6)
                b_geom_v = (b_geom_v - b_geom_v.min()) / (b_geom_v.max() - b_geom_v.min() + 1e-6)

                b_scores = self.w_feat * b_feat_c + self.w_geom * b_geom_v

                num_kept = max(self.min_keep, int(b_size * (1.0 - self.prune_ratio)))
                _, keep_idx = torch.topk(b_scores, num_kept)

                b_mask = torch.zeros(b_size, dtype=torch.bool, device=device)
                b_mask[keep_idx] = True
                keep_masks[start_idx:end_idx] = b_mask
            
            prev = new_offset[i - 1] if i > 0 else 0
            new_offset[i] = prev + num_kept
            start_idx = end_idx

        return keep_masks, new_offset


class GlobalContextToken(nn.Module):
    """
    Fully vectorized, loop-free GCT using Semantic Anchors and Gated Injection.
    """
    def __init__(self, channels: int, num_anchors: int = 4):
        super().__init__()
        self.channels = channels
        
        # Learnable semantic anchors
        self.semantic_anchors = nn.Parameter(torch.randn(num_anchors, channels) * (channels ** -0.5))
        
        # Projections
        self.pool_proj = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.gate = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.GELU(),
            nn.Linear(channels // 4, 1) # Scalar gate
        )
        self.out_proj = nn.Linear(channels, channels)
        
    def forward(self, point):
        feat = point.feat       # [N, C]
        offset = point.offset   # [B]
        N = feat.shape[0]
        B = offset.shape[0]
        
        # 1. Vectorized Batch Indexing (No Loops)
        batch_idx = torch.zeros(N, dtype=torch.long, device=feat.device)
        if B > 1:
            batch_idx[offset[:-1]] = 1
        batch_idx = torch.cumsum(batch_idx, dim=0) # [N]
        
        # 2. Vectorized Pooling (No Loops)
        mean_pool = scatter_mean(feat, batch_idx, dim=0, dim_size=B) # [B, C]
        max_pool, _ = scatter_max(feat, batch_idx, dim=0, dim_size=B) # [B, C]
        pooled = torch.cat([mean_pool, max_pool], dim=-1) # [B, 2C]
        
        # 3. Global Summary & Semantic Anchors
        global_summary = self.pool_proj(pooled) # [B, C]
        
        # Attention to anchors
        attn_weights = torch.softmax(
            torch.matmul(global_summary, self.semantic_anchors.T) / (self.channels ** 0.5), 
            dim=-1
        ) # [B, K]
        semantic_context = torch.matmul(attn_weights, self.semantic_anchors) # [B, C]
        
        # 4. Vectorized Expand & Gated Injection (No Loops)
        expanded_context = semantic_context[batch_idx] # [N, C]
        
        alpha = torch.sigmoid(self.gate(feat)) # [N, 1]
        point.feat = feat + alpha * self.out_proj(expanded_context)
        
        return point


class SerializationPositionalEncoding(nn.Module):
    """
    Serialization Positional Encoding (SPE) for PTv3
    
    Encodes the Z-order serialization index into a learnable 
    positional embedding and adds it to point features.
    
    Args:
        channels: Feature dimension (e.g., 32 for first encoder stage)
        hidden_dim: Hidden dimension of MLP (default: 16)
    """
    
    def __init__(self, channels, hidden_dim=16):
        super().__init__()
        
        self.channels = channels
        self.hidden_dim = hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, channels)
        )
        
        # Small initialization for stable training
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, features, serialized_order):
        """
        Args:
            features: (N, C) - point features after embedding
            serialized_order: (N,) - Z-order indices from PTv3
        
        Returns:
            (N, C) - features with positional encoding added
        """
        assert features.shape[0] == serialized_order.shape[0], \
        f"features ({features.shape[0]}) and order ({serialized_order.shape[0]}) must have same length"

        # Normalize order to [0, 1]
        order = serialized_order.float()
        order_max = order.max()
        
        if order_max > 0:
            order_norm = order / order_max
        else:
            order_norm = torch.zeros_like(order)
        
        order_norm = order_norm.unsqueeze(1)  # (N, 1)
        
        # Generate positional encoding
        pe = self.mlp(order_norm)  # (N, C)
        
        # Add to features (residual connection)
        return features + pe


class SPEStageWrapper(PointModule):
    """Wrapper برای اعمال SPE در ابتدای هر stage"""
    def __init__(self, spe_module, stage_idx):
        super().__init__()
        self.spe = spe_module
        self.stage_idx = stage_idx
    
    def forward(self, point):
        # order جدید بعد از pooling
        serialized_order = point.serialized_order[0]
        point.feat = self.spe(point.feat, serialized_order)
        return point
    
    
class GatedSkipConnectionCL(nn.Module):
    """
    Learns a soft gate to balance encoder (skip) and decoder features.

    Formula:
        alpha = sigmoid( MLP([enc_feat || dec_feat]) )
        out   = alpha * enc_feat + (1 - alpha) * dec_feat

    This is superior to simple addition because:
      - alpha=1 → trust encoder (fine details)
      - alpha=0 → trust decoder (semantic context)
      - alpha=0.5 → equal blend

    Args:
        enc_channels: feature dim of encoder skip connection
        dec_channels: feature dim of decoder (gating signal)
        hidden_dim:   intermediate MLP width (default: enc_channels // 2)
    """

    def __init__(self, enc_channels: int, dec_channels: int, hidden_dim: int = None):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = max(enc_channels // 2, 16)

        self.gate = nn.Sequential(
            nn.Linear(enc_channels + dec_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),   # LayerNorm: safe for batch_size=1
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, enc_channels),
            nn.Sigmoid(),
        )

        # A projection layer for dec_feat if channels don't match
        if enc_channels != dec_channels:
            self.dec_proj = nn.Linear(dec_channels, enc_channels)
        else:
            self.dec_proj = nn.Identity()

    def forward(self, enc_feat: torch.Tensor, dec_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            enc_feat: (N, enc_channels) — encoder skip features
            dec_feat: (N, dec_channels) — decoder gating signal

        Returns:
            (N, enc_channels) — gated blend of encoder and decoder
        """
        alpha = self.gate(torch.cat([enc_feat, dec_feat], dim=-1))  # (N, enc_channels)
        
        # Project dec_feat to match enc_feat's dimension
        dec_feat_proj = self.dec_proj(dec_feat)

        return alpha * enc_feat + (1.0 - alpha) * dec_feat_proj


class GatedSkipConnectionGE(nn.Module):
    """
    Gated Skip Connection based on Attention U-Net (Oktay et al., 2018)
    Adapted for Unstructured Point Clouds in PTv3.
    """
    def __init__(self, encoder_channels, decoder_channels, inter_channels=None):
        super().__init__()

        # برای کاهش مصرف VRAM، ابعاد واسط را نصف می‌کنیم
        inter_channels = inter_channels or max(min(encoder_channels, decoder_channels) // 2, 1)

        # تبدیلات خطی برای دیکودر (Gating Signal) و انکودر (Skip Connection)
        self.W_g = nn.Linear(decoder_channels, inter_channels, bias=False)
        self.W_x = nn.Linear(encoder_channels, inter_channels, bias=False)
        
        # تولید ضریب توجه (Attention Coefficient)
        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(inter_channels, 1, bias=True),
            nn.Sigmoid()
        )

        self.decoder_proj = (
            nn.Identity()
            if decoder_channels == encoder_channels
            else nn.Linear(decoder_channels, encoder_channels, bias=False)
        )
        
    def forward(self, encoder_feat, decoder_feat):
        # 1. انتقال ویژگی‌ها به فضای میانی مشترک
        g1 = self.W_g(decoder_feat)
        x1 = self.W_x(encoder_feat)
        
        # 2. ترکیب Additive (بسیار بهینه‌تر از Concat برای VRAM)
        psi = self.psi(g1 + x1)  # خروجی: ماتریسی با ابعاد (N, 1) حاوی اعداد بین 0 و 1
        
        decoder_feat = self.decoder_proj(decoder_feat)

        # 3. اعمال دروازه روی ویژگی‌های انکودر و جمع با دیکودر
        return (encoder_feat * psi) + decoder_feat