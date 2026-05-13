"""
train.py - segmentare binară: chela vs background
===================================================
UNet + EfficientNet-B4 pre-antrenat (ImageNet) cu:
  - BCE + Dice loss (binar)
  - mixed precision (--amp)
  - multe augmentări
  - early stopping

Instalare dependente:
    pip install segmentation-models-pytorch albumentations tqdm

Rulare:
    python train.py --data_dir dataset --epochs 100 --batch_size 4 --amp

    # Fără mixed precision (dacă GPU-ul nu suportă)
python train.py --data_dir dataset --epochs 100 --batch_size 4

# Batch size mai mic (dacă ai out of memory)
python train.py --data_dir dataset --epochs 100 --batch_size 2 --amp

# Learning rate diferit
python train.py --data_dir dataset --epochs 100 --batch_size 4 --amp --lr 0.0001

# Backbone diferit (mai ușor sau mai greu)
python train.py --data_dir dataset --epochs 100 --batch_size 4 --amp --backbone efficientnet-b0

# Early stopping mai agresiv (oprește după 10 epoci fără îmbunătățire)
python train.py --data_dir dataset --epochs 100 --batch_size 4 --amp --patience 10

# Salvare cu alt nume
python train.py --data_dir dataset --epochs 100 --batch_size 4 --amp --out_model my_model.pth
"""

import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast

import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import segmentation_models_pytorch as smp
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir",      default="dataset")
parser.add_argument("--epochs",        type=int,   default=100)
parser.add_argument("--batch_size",    type=int,   default=4)
parser.add_argument("--lr",            type=float, default=1e-4)
parser.add_argument("--img_size",      type=int,   default=512)
parser.add_argument("--backbone",      default="efficientnet-b4")
parser.add_argument("--out_model",     default="best_model.pth")
parser.add_argument("--amp",           action="store_true")  # Automatic Mixed Precision pentru float 16 ca GPU sa foloseasca mai putina memorie si sa fie mai rapid
parser.add_argument("--patience",      type=int,   default=15,
                    help="Early stopping: epoci fara imbunatatire")
args = parser.parse_args()

# ──────────────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# https://www.image-net.org/ Setul de date antrenat modelul EfficientNet-B4
# https://docs.pytorch.org/vision/stable/transforms.html stabilirea MEAN si STD pentru normalizare
# ──────────────────────────────────────────────────────────────────────────────
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

# Randomizez la fiecare epoca pentru a avea un antrenament mai robust
train_transform = A.Compose([
    A.Resize(args.img_size, args.img_size),

    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.25,
                       rotate_limit=45, border_mode=0, p=0.6),
    A.ElasticTransform(p=0.3),                 # deformare elastică
    A.GridDistortion(p=0.2),
    # Culoare
    A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1, p=0.6),
    A.GaussNoise(p=0.3),
    A.GaussianBlur(blur_limit=(3, 7), p=0.2),
    A.RandomShadow(p=0.2),                     # umbre artificiale
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(args.img_size, args.img_size),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])


class SegDataset(Dataset):
    def __init__(self, split: str, transform=None):
        self.img_dir  = Path(args.data_dir) / split / "images"
        self.mask_dir = Path(args.data_dir) / split / "masks"
        self.transform = transform

        self.pairs = []
        for img_path in sorted(self.img_dir.iterdir()):
            mask_path = self.mask_dir / (img_path.stem + ".png")
            if mask_path.exists():
                self.pairs.append((img_path, mask_path))

        print(f"  [{split}] {len(self.pairs)} imagini")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        img  = np.array(Image.open(img_path).convert("RGB"))
        # Masca e salvată cu 0/255 → o convertim la 0/1
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 127).astype(np.float32)   # 0.0 sau 1.0

        if self.transform:
            aug  = self.transform(image=img, mask=mask)
            img  = aug["image"]   # tensor (3, H, W)
            mask = aug["mask"]    # tensor (H, W)

        return img, mask.unsqueeze(0)   # mask: (1, H, W) pentru BCE

# ──────────────────────────────────────────────────────────────────────────────
# DataLoaders
# ──────────────────────────────────────────────────────────────────────────────
train_ds = SegDataset("train", transform=train_transform)
val_ds   = SegDataset("val",   transform=val_transform)

train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True,  num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                          shuffle=False, num_workers=0, pin_memory=True)

# ──────────────────────────────────────────────────────────────────────────────
# Model - binar: 1 clasă output, sigmoid implicit în loss
# ──────────────────────────────────────────────────────────────────────────────
model = smp.Unet(
    encoder_name=args.backbone,
    encoder_weights="imagenet",   # <-- weights pre-antrenate!
    in_channels=3,
    classes=1,                    # binar: o singura harta de probabilitate
    activation=None,              # logits brute, sigmoid in loss
)
model = model.to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"\nModel: UNet + {args.backbone} (imagenet) | {n_params:,} parametri")

# ──────────────────────────────────────────────────────────────────────────────
# Loss: BCE + Dice|F1 Score -pentru segmentare binara
# ──────────────────────────────────────────────────────────────────────────────
bce_loss  = nn.BCEWithLogitsLoss() #BCEWithLogistsLoss aplica sigmoid intern si calculeaza eroarea per pixel.
dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=True)

def combined_loss(logits, masks):
    return 0.5 * bce_loss(logits, masks) + 0.5 * dice_loss(logits, masks)

# ──────────────────────────────────────────────────────────────────────────────
# Optimizer - learning rate mic pentru encoder (e deja bun), mai mare pt decoder
# ──────────────────────────────────────────────────────────────────────────────
optimizer = torch.optim.AdamW([
    {"params": model.encoder.parameters(),          "lr": args.lr * 0.1}, # Encoder-ul e pre-antrenat -> learning rate mai mic ca sa nu strice ce a invatat deja
    {"params": model.decoder.parameters(),          "lr": args.lr},
    {"params": model.segmentation_head.parameters(),"lr": args.lr},
], weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=7, factor=0.5
)

scaler = GradScaler("cuda", enabled=args.amp)

# ──────────────────────────────────────────────────────────────────────────────
# Metrica: Dice / F1 pe validare
# ──────────────────────────────────────────────────────────────────────────────
def dice_score(logits, masks, threshold=0.5):
    preds = (torch.sigmoid(logits) > threshold).float()
    inter = (preds * masks).sum()
    union = preds.sum() + masks.sum()
    if union == 0:
        return 1.0
    return (2.0 * inter / union).item()

# ──────────────────────────────────────────────────────────────────────────────
# Training loop cu early stopping
# ──────────────────────────────────────────────────────────────────────────────
best_val_loss    = float("inf")
patience_counter = 0

for epoch in range(1, args.epochs + 1):

    # ── Train ──
    model.train()
    train_loss = 0.0
    for imgs, masks in tqdm(train_loader,
                            desc=f"Ep {epoch:3d}/{args.epochs} [Train]",
                            leave=False):
        imgs  = imgs.to(device,  non_blocking=True) #Mut batch-ul pe GPU
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast("cuda", enabled=args.amp):
            logits = model(imgs)
            loss   = combined_loss(logits, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # ── Val ──
    model.eval()
    val_loss = 0.0
    val_dice = 0.0
    with torch.no_grad(): # Dezactivez calculul gradientilor. Nu am nevoie la validare.
        for imgs, masks in tqdm(val_loader,
                                desc=f"Ep {epoch:3d}/{args.epochs} [Val]  ",
                                leave=False):
            imgs  = imgs.to(device,  non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with autocast("cuda", enabled=args.amp):
                logits = model(imgs)
                loss   = combined_loss(logits, masks)
            val_loss += loss.item()
            val_dice += dice_score(logits, masks)

    val_loss /= len(val_loader)
    val_dice /= len(val_loader)

    scheduler.step(val_loss) # ReduceLROnPlateau ajustează learning rate-ul dacă val_loss nu se îmbunătățește

    marker = ""
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save({
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "val_loss":    val_loss,
            "val_dice":    val_dice,
            "backbone":    args.backbone,
            "classes":     ["background", "chela"],
        }, args.out_model)
        marker = "  salvat"
    else:
        patience_counter += 1

    print(f"Ep {epoch:3d} | Train: {train_loss:.4f} | "
          f"Val: {val_loss:.4f} | Dice: {val_dice:.4f}{marker}")

    if patience_counter >= args.patience:
        print(f"\nEarly stopping la epoca {epoch} (fara imbunatatire in {args.patience} epoci)")
        break

print(f"\nBest val loss: {best_val_loss:.4f} → {args.out_model}")