# test_ptv3_cpu.py
"""
اسکریپت تست و trace کامل برای Point Transformer V3
این اسکریپت شامل:
1. Mock کردن فقط وابستگی‌های CUDA-only
2. تست forward pass
3. تست backward pass
4. نمایش جزئیات هر مرحله برای trace و debug
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

print("=" * 80)
print("Point Transformer V3 - CPU Test & Trace Script")
print("=" * 80)

# ============================================================================
# بخش 1: Mock کردن فقط وابستگی‌های CUDA-only
# ============================================================================
print("\n[بخش 1] Mock کردن وابستگی‌های CUDA-only...")

from mock_dependencies import install_all_mocks
install_all_mocks()

print("\n✓ Mock های CUDA-only آماده شدند")
print("✓ وابستگی‌های CPU-compatible از نسخه واقعی استفاده می‌کنند")

# حالا torch و سایر کتابخانه‌ها رو import می‌کنیم
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# بخش 2: Import مدل
# ============================================================================
print("\n[بخش 2] Import مدل...")
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import PointTransformerV3
print("✓ PointTransformerV3 import شد")

# ============================================================================
# بخش 3: تابع ساخت داده ساختگی
# ============================================================================
def create_dummy_data(num_points=2048, num_classes=13, verbose=True):
    """ساخت داده ساختگی واقع‌گرایانه"""
    if verbose:
        print(f"\n[ساخت داده] تعداد نقاط: {num_points}, تعداد کلاس‌ها: {num_classes}")
    
    # مختصات 3D تصادفی
    coord = torch.randn(num_points, 3) * 10
    if verbose:
        print(f"  • coord: {coord.shape} - range: [{coord.min():.2f}, {coord.max():.2f}]")
    
    # ویژگی‌ها (مثلاً RGB)
    feat = torch.randn(num_points, 3)
    if verbose:
        print(f"  • feat: {feat.shape}")
    
    # Grid coordinates برای sparse convolution
    grid_size = 0.02
    grid_coord = (coord / grid_size).long()
    grid_coord = grid_coord - grid_coord.min(0)[0]
    if verbose:
        print(f"  • grid_coord: {grid_coord.shape} - grid_size: {grid_size}")
    
    # برچسب‌های segmentation
    segment = torch.randint(0, num_classes, (num_points,))
    if verbose:
        print(f"  • segment: {segment.shape} - classes: {segment.unique().tolist()}")
    
    # Offset برای batch processing
    offset = torch.tensor([num_points], dtype=torch.long)
    if verbose:
        print(f"  • offset: {offset.shape}")
    
    return {
        'coord': coord,
        'feat': feat,
        'grid_coord': grid_coord,
        'segment': segment,
        'offset': offset,
    }

# ============================================================================
# بخش 4: تابع تست کامل
# ============================================================================
def test_full_pipeline():
    print("\n" + "=" * 80)
    print("شروع تست کامل Pipeline")
    print("=" * 80)
    
    # کانفیگ مدل
    model_config = {
        'in_channels': 3,
        'order': ('z', 'z-trans'),
        'stride': (2, 2, 2, 2),
        'enc_depths': (2, 2, 2, 6, 2),
        'enc_channels': (32, 64, 128, 256, 512),
        'enc_num_head': (2, 4, 8, 16, 32),
        'enc_patch_size': (48, 48, 48, 48, 48),
        'dec_depths': (2, 2, 2, 2),
        'dec_channels': (64, 64, 128, 256),
        'dec_num_head': (4, 4, 8, 16),
        'dec_patch_size': (48, 48, 48, 48),
        'mlp_ratio': 4,
        'qkv_bias': True,
        'qk_scale': None,
        'attn_drop': 0.0,
        'proj_drop': 0.0,
        'drop_path': 0.3,
        'pre_norm': True,
        'shuffle_orders': True,
        'enable_rpe': False,
        'enable_flash': False,
        'upcast_attention': False,
        'upcast_softmax': False,
        'pdnorm_bn': False,
        'pdnorm_ln': False,
        'pdnorm_decouple': True,
        'pdnorm_adaptive': False,
    }
    
    num_classes = 13
    
    try:
        # مرحله 1: ساخت مدل
        print("\n[مرحله 1/6] ساخت مدل...")
        print("  کانفیگ:")
        for key, value in list(model_config.items())[:5]:
            print(f"    - {key}: {value}")
        print("    ...")
        
        model = PointTransformerV3(**model_config)
        model.train()
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n  ✓ مدل ساخته شد:")
        print(f"    - کل پارامترها: {total_params:,}")
        print(f"    - پارامترهای قابل آموزش: {trainable_params:,}")
        
        # مرحله 2: اضافه کردن segmentation head
        print("\n[مرحله 2/6] اضافه کردن Segmentation Head...")
        final_channels = model_config['dec_channels'][0]
        seg_head = nn.Linear(final_channels, num_classes)
        print(f"  ✓ Head اضافه شد: Linear({final_channels} → {num_classes})")
        
        # مرحله 3: ساخت داده
        print("\n[مرحله 3/6] ساخت داده ساختگی...")
        data_dict = create_dummy_data(num_points=2048, num_classes=num_classes, verbose=True)
        
        # مرحله 4: Forward pass
        print("\n[مرحله 4/6] Forward Pass...")
        print("  → اجرای model(data_dict)...")
        output = model(data_dict)
        
        if isinstance(output, dict) and 'feat' in output:
            features = output['feat']
            print(f"  ✓ خروجی به صورت dict با کلید 'feat'")
        elif isinstance(output, torch.Tensor):
            features = output
            print(f"  ✓ خروجی به صورت tensor")
        else:
            raise ValueError(f"نوع خروجی غیرمنتظره: {type(output)}")
        
        print(f"  • Features shape: {features.shape}")
        print(f"  • Features range: [{features.min():.4f}, {features.max():.4f}]")
        print(f"  • Features mean: {features.mean():.4f}, std: {features.std():.4f}")
        
        print("\n  → اعمال segmentation head...")
        logits = seg_head(features)
        print(f"  ✓ Logits shape: {logits.shape}")
        print(f"  • Logits range: [{logits.min():.4f}, {logits.max():.4f}]")
        
        # مرحله 5: محاسبه Loss
        print("\n[مرحله 5/6] محاسبه Loss...")
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, data_dict['segment'])
        print(f"  ✓ Loss: {loss.item():.6f}")
        
        # مرحله 6: Backward pass
        print("\n[مرحله 6/6] Backward Pass...")
        print("  → اجرای loss.backward()...")
        loss.backward()
        print("  ✓ Backward موفق بود")
        
        # بررسی gradients
        has_grad = sum(1 for p in model.parameters() if p.grad is not None)
        total_params_count = sum(1 for _ in model.parameters())
        print(f"\n  • Gradients محاسبه شده: {has_grad}/{total_params_count} parameters")
        
        # نمایش نمونه gradient
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"  • نمونه gradient - {name}:")
                print(f"    - shape: {param.grad.shape}")
                print(f"    - mean: {param.grad.mean():.6f}, std: {param.grad.std():.6f}")
                break
        
        # خلاصه نهایی
        print("\n" + "=" * 80)
        print("✓✓✓ تست کامل با موفقیت انجام شد! ✓✓✓")
        print("=" * 80)
        print("\nخلاصه:")
        print(f"  ✓ مدل: {total_params:,} پارامتر")
        print(f"  ✓ داده: {data_dict['coord'].shape[0]} نقطه")
        print(f"  ✓ Forward: features {features.shape}")
        print(f"  ✓ Loss: {loss.item():.6f}")
        print(f"  ✓ Backward: {has_grad}/{total_params_count} gradients")
        print("\n→ کد شما آماده است برای:")
        print("  • تست با داده واقعی")
        print("  • Training loop")
        print("  • Evaluation")
        
        return True
        
    except Exception as e:
        print(f"\n" + "=" * 80)
        print("✗✗✗ خطا رخ داد ✗✗✗")
        print("=" * 80)
        print(f"\nنوع خطا: {type(e).__name__}")
        print(f"پیام: {str(e)}")
        print("\nTraceback کامل:")
        import traceback
        traceback.print_exc()
        print("\n→ این خطا باید قبل از ادامه کار رفع شود")
        return False

# ============================================================================
# اجرای اصلی
# ============================================================================
if __name__ == "__main__":
    success = test_full_pipeline()
    
    print("\n" + "=" * 80)
    if success:
        print("وضعیت نهایی: موفق ✓")
    else:
        print("وضعیت نهایی: ناموفق ✗")
    print("=" * 80)
