#!/usr/bin/env python3
"""
AGGRESSIVE BALANCED TRAINING - Ultra-focused on lesion detection
Using extreme techniques to force the model to learn lesions
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import time
import gc
import random

# Import MONAI components
import monai
from monai.networks.nets import UNet
from monai.losses import DiceLoss, TverskyLoss, FocalLoss
from monai.data import Dataset, DataLoader, CacheDataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, RandCropByPosNegLabeld, ResizeWithPadOrCropd, 
    RandFlipd, RandRotate90d, RandShiftIntensityd, EnsureTyped, ToTensord,
    CropForegroundd
)
import nibabel as nib

def test_imports():
    """Test all required imports"""
    try:
        print("✅ PyTorch imported successfully")
        print("✅ MONAI imported successfully")
        print("✅ nibabel imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def get_aggressive_data(max_samples=50):
    """Get data with focus on samples that have substantial lesions"""
    print(f"📊 Loading ATLAS dataset with AGGRESSIVE lesion selection (max {max_samples} samples)...")
    
    atlas_dir = Path("ATLAS_2")
    train_dir = atlas_dir / "Training"
    
    if not train_dir.exists():
        print(f"❌ Training directory not found: {train_dir}")
        return [], []
    
    train_files = []
    sample_count = 0
    
    # Scan for data and prioritize samples with larger lesions
    lesion_samples = []
    
    for site_dir in train_dir.glob("R*"):
        if site_dir.is_dir():
            for subject_dir in site_dir.glob("sub-*"):
                if subject_dir.is_dir():
                    session_dir = subject_dir / "ses-1"
                    if session_dir.exists():
                        anat_dir = session_dir / "anat"
                        if anat_dir.exists():
                            t1w_file = None
                            mask_file = None
                            
                            for file in anat_dir.iterdir():
                                if file.suffix == '.gz' and file.name.endswith('.nii.gz'):
                                    if 'T1w' in file.name and 'lesion' not in file.name:
                                        t1w_file = file
                                    elif 'lesion' in file.name or 'mask' in file.name:
                                        mask_file = file
                            
                            if t1w_file and mask_file:
                                # Quick check of lesion volume
                                try:
                                    mask_img = nib.load(str(mask_file))
                                    mask_data = mask_img.get_fdata()
                                    lesion_volume = np.sum(mask_data > 0)
                                    
                                    if lesion_volume > 500:  # Only use samples with substantial lesions
                                        lesion_samples.append({
                                            "image": str(t1w_file),
                                            "label": str(mask_file),
                                            "subject": subject_dir.name,
                                            "site": site_dir.name,
                                            "lesion_volume": lesion_volume
                                        })
                                        print(f"   Found lesion sample: {subject_dir.name} (volume: {lesion_volume})")
                                        
                                except Exception as e:
                                    continue
    
    # Sort by lesion volume (largest first) and take the best samples
    lesion_samples.sort(key=lambda x: x["lesion_volume"], reverse=True)
    train_files = lesion_samples[:max_samples]
    
    print(f"   Selected {len(train_files)} high-lesion samples")
    print(f"   Lesion volumes range: {train_files[-1]['lesion_volume']} - {train_files[0]['lesion_volume']}")
    
    # Simple 80/20 split
    random.shuffle(train_files)
    split_idx = int(0.8 * len(train_files))
    train_data = train_files[:split_idx]
    val_data = train_files[split_idx:]
    
    print(f"   Training samples: {len(train_data)}")
    print(f"   Validation samples: {len(val_data)}")
    
    return train_data, val_data

def create_aggressive_transforms():
    """Create transforms focused entirely on lesion regions"""
    train_transform = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 1.5), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        
        # AGGRESSIVE: Crop around lesions with high probability
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(64, 64, 64),  # Smaller patches for better lesion focus
            pos=10,  # MUCH more positive samples
            neg=1,   # Very few negative samples
            num_samples=4,  # More patches per sample
        ),
        
        # Data augmentation focused on preserving lesions
        RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.8),
        RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.8),
        RandRotate90d(keys=["image", "label"], prob=0.8, max_k=3),
        RandShiftIntensityd(keys=["image"], offsets=0.2, prob=0.8),
        
        EnsureTyped(keys=["image", "label"]),
        ToTensord(keys=["image", "label"])
    ])
    
    val_transform = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 1.5), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(96, 96, 96)),
        EnsureTyped(keys=["image", "label"]),
        ToTensord(keys=["image", "label"])
    ])
    
    return train_transform, val_transform

class AggressiveLoss(nn.Module):
    """Ultra-aggressive loss function heavily weighted towards lesions"""
    def __init__(self, device='cuda'):
        super().__init__()
        # EXTREME positive weight for lesions
        pos_weight = torch.tensor([500.0]).to(device)  # 500x weight for lesions!
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss(sigmoid=True)
        self.tversky = TverskyLoss(alpha=0.1, beta=0.9)  # Heavy recall focus
        self.focal = FocalLoss(gamma=3.0)  # High gamma for hard examples
    
    def forward(self, pred, target):
        # Ensure shapes match
        if pred.shape != target.shape:
            pred = pred.reshape(target.shape)
        
        # Binary classification
        pred_flat = pred.view(-1)
        target_flat = target.view(-1).float()
        
        # Multiple loss components with heavy lesion weighting
        bce_loss = self.bce(pred_flat, target_flat)
        dice_loss = self.dice(pred, target)
        tversky_loss = self.tversky(pred, target)
        focal_loss = self.focal(pred, target)
        
        # Heavily weighted combination
        total_loss = 2.0 * bce_loss + 3.0 * dice_loss + 2.0 * tversky_loss + 1.0 * focal_loss
        
        return total_loss

def dice_score(pred, target, smooth=1e-6):
    """Calculate Dice score"""
    pred = torch.sigmoid(pred)
    pred = (pred > 0.3).float()  # Lower threshold for more sensitive detection
    target = target.float()
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice.item()

def train_aggressive_model():
    """Train with aggressive lesion-focused approach"""
    print("🚀 AGGRESSIVE LESION-FOCUSED TRAINING")
    print("="*70)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 Device: {device}")
    
    # Get high-lesion data
    train_data, val_data = get_aggressive_data(max_samples=50)
    if not train_data:
        print("❌ No training data found!")
        return
    
    # Create transforms
    train_transform, val_transform = create_aggressive_transforms()
    
    # Create datasets with heavy caching
    train_dataset = CacheDataset(
        data=train_data,
        transform=train_transform,
        cache_rate=0.5,  # Cache more for speed
        num_workers=2
    )
    
    val_dataset = Dataset(
        data=val_data,
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,  # Larger batch size with smaller patches
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    
    print(f"🎯 AGGRESSIVE Training Configuration:")
    print(f"   Training samples: {len(train_data)} (high-lesion only)")
    print(f"   Validation samples: {len(val_data)}")
    print(f"   Batch size: 4")
    print(f"   Patch size: 64x64x64")
    print(f"   Lesion:Background ratio: 10:1")
    print(f"   Loss weight: 500x for lesions")
    
    # Create smaller, faster model
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(8, 16, 32, 64, 128),  # Smaller model
        strides=(2, 2, 2, 2),
        num_res_units=1,  # Faster training
    ).to(device)
    
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup training with aggressive settings
    criterion = AggressiveLoss(device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)  # Higher LR
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)
    
    # Training loop
    best_dice = 0.0
    epochs = 20  # More epochs
    
    print(f"\n🔄 Starting AGGRESSIVE training for {epochs} epochs...")
    
    for epoch in range(epochs):
        print(f"\n📈 Epoch {epoch+1}/{epochs}")
        
        # Training
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        train_batches = 0
        
        for batch_idx, batch_data in enumerate(train_loader):
            images = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_dice += dice_score(outputs, labels)
            train_batches += 1
            
            if batch_idx % 5 == 0:
                print(f"   Batch {batch_idx+1}: Loss={loss.item():.4f}")
        
        avg_train_loss = train_loss / train_batches
        avg_train_dice = train_dice / train_batches
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_data in val_loader:
                images = batch_data["image"].to(device)
                labels = batch_data["label"].to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                val_dice += dice_score(outputs, labels)
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
        avg_val_dice = val_dice / val_batches if val_batches > 0 else 0
        
        print(f"   Train - Loss: {avg_train_loss:.4f}, Dice: {avg_train_dice:.4f}")
        print(f"   Val   - Loss: {avg_val_loss:.4f}, Dice: {avg_val_dice:.4f}")
        
        # Save best model
        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            torch.save(model.state_dict(), "stroke_segment_aggressive.pth")
            print(f"   💾 New best model saved! Dice: {best_dice:.4f}")
        
        scheduler.step(avg_val_dice)
        
        # Clean up GPU memory
        torch.cuda.empty_cache()
        gc.collect()
    
    print(f"\n🎉 AGGRESSIVE training completed!")
    print(f"   Best validation Dice: {best_dice:.4f}")
    print(f"   Model saved as: stroke_segment_aggressive.pth")
    
    return model

if __name__ == "__main__":
    if not test_imports():
        sys.exit(1)
    
    model = train_aggressive_model()
