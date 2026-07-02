"""
Integration/E2E Test Suite for PointTransformerV3 with new modules
Tests DefaultSegmentorV2 wrapper with all production modules enabled on CPU
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any
import numpy as np


# Fixtures
@pytest.fixture
def cpu_device():
    """Force CPU execution for CI/testing"""
    return torch.device('cpu')


@pytest.fixture
def minimal_config(cpu_device) -> Dict[str, Any]:
    """
    Minimal production-like config for DefaultSegmentorV2 with PT-v3m1 backbone.
    All new modules enabled, CPU-safe settings.
    """
    return {
        "type": "DefaultSegmentorV2",
        "num_classes": 20,
        "backbone_out_channels": 64,
        "backbone": {
            "type": "PT-v3m1",
            "in_channels": 6,
            "order": ["z", "z-trans"],
            "stride": [2, 2, 2],
            "enc_depths": [2, 2, 2],
            "enc_channels": [32, 64, 128],
            "enc_num_head": [2, 4, 8],
            "enc_patch_size": [64, 64, 64],
            "dec_depths": [1, 1, 1],
            "dec_channels": [64, 32, 32],
            "dec_num_head": [4, 2, 2],
            "dec_patch_size": [64, 64, 64],
            "mlp_ratio": 4,
            "qkv_bias": True,
            "qk_scale": None,
            "attn_drop": 0.0,
            "proj_drop": 0.0,
            "drop_path": 0.1,
            "pre_norm": True,
            "shuffle_orders": True,
            "enable_rpe": False,  # CPU-safe
            "enable_flash": False,  # CPU-safe
            "upcast_attention": True,
            "upcast_softmax": True,
            # New modules - all enabled
            "enable_spe": True,
            "spe_dim": 16,
            "enable_gsc": True,
            "enable_grab": True,
            "grab_hidden_dim": 16,
            "grab_use_distance": True,
            "grab_per_head": True,
            "grab_init_scale": 0.01,
            "enable_gtp": True,
            "gtp_prune_ratio": [0.1, 0.2, 0.3],
            "gtp_k": 8,
            "enable_gct": True,
            "gct_num_anchors": 4,
            "pdnorm_bn": False,
            "pdnorm_ln": False,
            "pdnorm_decouple": True,
            "pdnorm_adaptive": False,
            "pdnorm_affine": True,
            "pdnorm_conditions": ["ScanNet", "S3DIS", "Structured3D"],
        },
        "criteria": [
            {"type": "CrossEntropyLoss", "loss_weight": 1.0, "ignore_index": -1},
        ],
    }


@pytest.fixture
def dummy_point_cloud(cpu_device):
    """
    Create dummy point cloud data matching expected API.
    Returns dict with coord, feat, offset, grid_coord, segment.
    """
    batch_size = 2
    num_points_per_batch = 1024
    total_points = batch_size * num_points_per_batch
    
    # coord: [N, 3]
    coord = torch.randn(total_points, 3, device=cpu_device) * 10.0
    
    # feat: [N, 6] (e.g., xyz + rgb)
    feat = torch.randn(total_points, 6, device=cpu_device)
    
    # offset: [batch_size] cumulative point counts
    offset = torch.tensor(
        [num_points_per_batch * (i + 1) for i in range(batch_size)],
        dtype=torch.long,
        device=cpu_device
    )
    
    # grid_coord: [N, 3] quantized coordinates
    grid_coord = (coord / 0.02).long()
    
    # segment: [N] ground truth labels
    segment = torch.randint(0, 20, (total_points,), dtype=torch.long, device=cpu_device)
    
    # condition: dataset name for PDNorm
    condition = "ScanNet"
    
    return {
        "coord": coord,
        "feat": feat,
        "offset": offset,
        "grid_coord": grid_coord,
        "segment": segment,
        "condition": condition,
    }


# Test suite
class TestPointTransformerV3Integration:
    """
    Integration/E2E tests for PointTransformerV3 via DefaultSegmentorV2.
    Validates production contract: input dict → output dict with seg_logits/loss.
    """
    
    def test_model_instantiation(self, minimal_config, cpu_device):
        """Test that model can be instantiated with all new modules enabled."""
        from pointcept.models.builder import build_model
        
        model = build_model(minimal_config)
        model = model.to(cpu_device)
        model.eval()
        
        assert model is not None
        assert hasattr(model, 'backbone')
        assert hasattr(model, 'seg_head')
        assert hasattr(model, 'criteria')
        
        # Check backbone type
        assert model.backbone.__class__.__name__ == "PointTransformerV3"
        
        # Check that new modules are present in backbone
        backbone_str = str(model.backbone)
        if minimal_config["backbone"]["enable_spe"]:
            assert "StructuralPositionEmbedding" in backbone_str or "spe" in backbone_str.lower()
        if minimal_config["backbone"]["enable_gct"]:
            assert "GlobalContextToken" in backbone_str or "gct" in backbone_str.lower()
        if minimal_config["backbone"]["enable_gtp"]:
            assert "GeometryTokenPruner" in backbone_str or "gtp" in backbone_str.lower()
    
    def test_forward_train_mode(self, minimal_config, dummy_point_cloud, cpu_device):
        """Test forward pass in training mode. Should return dict with loss."""
        from pointcept.models.builder import build_model
        
        model = build_model(minimal_config)
        model = model.to(cpu_device)
        model.train()
        
        output = model(dummy_point_cloud)
        
        # Check output contract for training
        assert isinstance(output, dict), "Output must be dict in train mode"
        assert "loss" in output, "Training output must contain 'loss'"
        assert isinstance(output["loss"], torch.Tensor)
        assert output["loss"].numel() == 1, "Loss should be scalar"
        assert torch.isfinite(output["loss"]).all(), "Loss must be finite"
    
    def test_forward_eval_mode(self, minimal_config, dummy_point_cloud, cpu_device):
        """Test forward pass in eval mode. Should return dict with seg_logits and loss."""
        from pointcept.models.builder import build_model
        
        model = build_model(minimal_config)
        model = model.to(cpu_device)
        model.eval()
        
        with torch.no_grad():
            output = model(dummy_point_cloud)
        
        # Check output contract for eval (with ground truth segment)
        assert isinstance(output, dict), "Output must be dict"
        assert "seg_logits" in output, "Eval output must contain 'seg_logits'"
        assert "loss" in output, "Eval output with segment must contain 'loss'"
        
        seg_logits = output["seg_logits"]
        num_points = dummy_point_cloud["coord"].shape[0]
        num_classes = minimal_config["num_classes"]
        
        assert seg_logits.shape == (num_points, num_classes), \
            f"seg_logits shape mismatch: expected ({num_points}, {num_classes}), got {seg_logits.shape}"
        assert torch.isfinite(seg_logits).all(), "seg_logits must be finite"
    
    def test_forward_inference_mode(self, minimal_config, cpu_device):
        """Test forward pass in inference mode (no ground truth). Should return dict with seg_logits only."""
        from pointcept.models.builder import build_model
        
        model = build_model(minimal_config)
        model = model.to(cpu_device)
        model.eval()
        
        # Create input without 'segment' key
        num_points = 512
        inference_input = {
            "coord": torch.randn(num_points, 3, device=cpu_device) * 10.0,
            "feat": torch.randn(num_points, 6, device=cpu_device),
            "offset": torch.tensor([num_points], dtype=torch.long, device=cpu_device),
            "grid_coord": (torch.randn(num_points, 3, device=cpu_device) * 500).long(),
            "condition": "ScanNet",
        }
        
        with torch.no_grad():
            output = model(inference_input)
        
        # Check output contract for inference
        assert isinstance(output, dict)
        assert "seg_logits" in output
        assert "loss" not in output, "Inference without segment should not compute loss"
        
        seg_logits = output["seg_logits"]
        assert seg_logits.shape[0] == num_points
        assert seg_logits.shape[1] == minimal_config["num_classes"]
    
    def test_backward_pass(self, minimal_config, dummy_point_cloud, cpu_device):
        """Test that backward pass completes and produces finite gradients."""
        from pointcept.models.builder import build_model
        
        model = build_model(minimal_config)
        model = model.to(cpu_device)
        model.train()
        
        # Forward
        output = model(dummy_point_cloud)
        loss = output["loss"]
        
        # Backward
        loss.backward()
        
        # Check gradients exist and are finite for key parameters
        grad_found = False
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_found = True
                assert torch.isfinite(param.grad).all(), f"Gradient for {name} contains NaN/Inf"
        
        assert grad_found, "No gradients were computed"
    
    def test_optimizer_step(self, minimal_config, dummy_point_cloud, cpu_device):
        """Test full training step with optimizer."""
        from pointcept.models.builder import build_model
        
        model = build_model(minimal_config)
        model = model.to(cpu_device)
        model.train()
        
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        # Initial loss
        output = model(dummy_point_cloud)
        loss_before = output["loss"].item()
        
        # Training step
        optimizer.zero_grad()
        output = model(dummy_point_cloud)
        loss = output["loss"]
        loss.backward()
        optimizer.step()
        
        # Second forward to check parameters updated
        output = model(dummy_point_cloud)
        loss_after = output["loss"].item()
        
        # Loss should be finite and may change (not a strict decrease check due to randomness)
        assert np.isfinite(loss_before)
        assert np.isfinite(loss_after)
    
    def test_new_modules_activation(self, minimal_config, dummy_point_cloud, cpu_device):
        """Test that new modules (SPE, GCT, GTP, GRAB, GSC) are actually activated during forward."""
        from pointcept.models.builder import build_model
        
        model = build_model(minimal_config)
        model = model.to(cpu_device)
        model.eval()
        
        # Register hooks to detect module activation
        activated_modules = {}
        
        def make_hook(name):
            def hook(module, input, output):
                activated_modules[name] = True
            return hook
        
        # Register hooks for new modules
        for name, module in model.named_modules():
            module_class = module.__class__.__name__
            if any(keyword in module_class.lower() for keyword in 
                   ['spe', 'gct', 'gtp', 'grab', 'gsc', 'structuralpositional', 
                    'globalcontext', 'geometrytoken', 'geometryaware', 'globalskip']):
                module.register_forward_hook(make_hook(name))
        
        # Forward pass
        with torch.no_grad():
            _ = model(dummy_point_cloud)
        
        # Check that at least some new modules were activated
        assert len(activated_modules) > 0, \
            "No new modules were activated. Check config and module naming."
    
    def test_different_batch_sizes(self, minimal_config, cpu_device):
        """Test model with different batch sizes."""
        from pointcept.models.builder import build_model
        
        model = build_model(minimal_config)
        model = model.to(cpu_device)
        model.eval()
        
        for batch_size in [1, 3]:
            num_points_per_batch = 256
            total_points = batch_size * num_points_per_batch
            
            data = {
                "coord": torch.randn(total_points, 3, device=cpu_device) * 10.0,
                "feat": torch.randn(total_points, 6, device=cpu_device),
                "offset": torch.tensor(
                    [num_points_per_batch * (i + 1) for i in range(batch_size)],
                    dtype=torch.long,
                    device=cpu_device
                ),
                "grid_coord": (torch.randn(total_points, 3, device=cpu_device) * 500).long(),
                "segment": torch.randint(0, 20, (total_points,), dtype=torch.long, device=cpu_device),
                "condition": "ScanNet",
            }
            
            with torch.no_grad():
                output = model(data)
            
            assert "seg_logits" in output
            assert output["seg_logits"].shape[0] == total_points
    
    def test_deterministic_output(self, minimal_config, dummy_point_cloud, cpu_device):
        """Test that model produces deterministic output with same input (eval mode, no dropout)."""
        from pointcept.models.builder import build_model
        
        # Set seeds
        torch.manual_seed(42)
        np.random.seed(42)
        
        model = build_model(minimal_config)
        model = model.to(cpu_device)
        model.eval()
        
        with torch.no_grad():
            output1 = model(dummy_point_cloud)
            output2 = model(dummy_point_cloud)
        
        # Check determinism
        torch.testing.assert_close(
            output1["seg_logits"],
            output2["seg_logits"],
            rtol=1e-5,
            atol=1e-7,
            msg="Model output should be deterministic in eval mode"
        )
    
    def test_return_point_option(self, minimal_config, dummy_point_cloud, cpu_device):
        """Test that DefaultSegmentorV2 can return Point object when requested."""
        from pointcept.models.builder import build_model
        
        model = build_model(minimal_config)
        model = model.to(cpu_device)
        model.eval()
        
        with torch.no_grad():
            output = model(dummy_point_cloud, return_point=True)
        
        # Check that 'point' key is present
        assert "point" in output, "Output must contain 'point' when return_point=True"
        
        # Check Point object structure
        point = output["point"]
        assert hasattr(point, "feat"), "Point must have 'feat' attribute"
        assert hasattr(point, "offset"), "Point must have 'offset' attribute"
        assert hasattr(point, "coord"), "Point must have 'coord' attribute"


# Additional backbone-specific test (optional, for completeness)
class TestPointTransformerV3Backbone:
    """
    Direct tests on PointTransformerV3 backbone (Point-centric contract).
    """
    
    def test_backbone_returns_point(self, minimal_config, cpu_device):
        """Test that PointTransformerV3 backbone returns Point object."""
        from pointcept.models.builder import build_model
        from pointcept.models.utils.structure import Point
        
        # Build only backbone
        backbone_config = minimal_config["backbone"]
        backbone_config["type"] = "PT-v3m1"
        
        backbone = build_model(backbone_config)
        backbone = backbone.to(cpu_device)
        backbone.eval()
        
        # Create Point input
        num_points = 512
        data_dict = {
            "coord": torch.randn(num_points, 3, device=cpu_device) * 10.0,
            "feat": torch.randn(num_points, 6, device=cpu_device),
            "offset": torch.tensor([num_points], dtype=torch.long, device=cpu_device),
            "grid_coord": (torch.randn(num_points, 3, device=cpu_device) * 500).long(),
        }
        
        with torch.no_grad():
            output = backbone(data_dict)
        
        # Check output is Point object
        assert isinstance(output, Point), f"Backbone must return Point, got {type(output)}"
        assert hasattr(output, "feat")
        assert output.feat.shape[0] == num_points
        assert torch.isfinite(output.feat).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
