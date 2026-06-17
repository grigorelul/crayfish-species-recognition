"""
TrainClassifier.py — Clasificare Raci de Apa Dulce
====================================================
AlexNet cu transfer learning din ImageNet.
Surse:
  [1] Krizhevsky et al. (2012) - ImageNet Classification with Deep CNNs
  [2] https://cvml.ista.ac.at/courses/DLWT_W17/material/AlexNet.pdf
"""

import random
import time
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

# ── Cai ──────────────────────────────────────────────────────────────────────

ROOT_DIR    = Path(__file__).resolve().parent
CROPPED_DIR = ROOT_DIR / "BazaDeDateRaciCropped"

CLASS_NAMES = {
    "Austropotamobius bihariensis": "Austropotamobius_bihariensis",
    "Austropotamobius torrentium":  "Austropotamobius_torrentium",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# ── Configurare ───────────────────────────────────────────────────────────────

IMG_SIZE        = 227
BATCH_SIZE      = 64
LEARNING_RATE   = 0.001
MOMENTUM        = 0.9
WEIGHT_DECAY    = 0.0005
NUM_EPOCHS      = 60
LABEL_SMOOTHING = 0.1

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ── Dataset PyTorch ───────────────────────────────────────────────────────────

class RaciDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ── Colectare imagini ─────────────────────────────────────────────────────────

def build_samples():
    samples   = []
    label_map = {name: i for i, name in enumerate(CLASS_NAMES.values())}

    for name, label_idx in label_map.items():
        cls_dir = CROPPED_DIR / name
        if not cls_dir.exists():
            print(f"Folder lipsa: {cls_dir}")
            continue
        for sub in ["complet", "truncated"]:
            sub_dir = cls_dir / sub
            if sub_dir.exists():
                for f in sorted(sub_dir.iterdir()):
                    if f.suffix.lower() in IMAGE_EXTENSIONS:
                        samples.append((str(f), label_idx))
        for f in sorted(cls_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((str(f), label_idx))

    if not samples:
        raise SystemExit(f"[EROARE] Nicio imagine gasita in {CROPPED_DIR}. Ruleaza CropRaci.py mai intai.")

    random.shuffle(samples)
    return samples, label_map


# ── Model: AlexNet cu transfer learning ──────────────────────────────────────

def build_alexnet_transfer(num_classes=2):
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)

    for param in model.features.parameters():
        param.requires_grad = False

    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)
    return model


# ── Antrenare ─────────────────────────────────────────────────────────────────

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispozitiv: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=45),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
        transforms.RandomResizedCrop(size=IMG_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    samples, label_map = build_samples()
    print(f"\nTotal imagini: {len(samples)} | Clase: {label_map}")

    per_class = {}
    for path, lbl in samples:
        per_class.setdefault(lbl, []).append((path, lbl))
    for lbl in per_class:
        random.shuffle(per_class[lbl]) # amestec imaginile din fiecare clasa

    train_samples, val_samples, test_samples = [], [], []
    class_idx_to_name = {v: k for k, v in label_map.items()}

    print("\nSplit per clasa:")
    for lbl, cls_samples in sorted(per_class.items()):
        n    = len(cls_samples)
        n_tr = int(n * TRAIN_RATIO)
        n_v  = int(n * VAL_RATIO)
        tr   = cls_samples[:n_tr]
        v    = cls_samples[n_tr:n_tr + n_v]
        te   = cls_samples[n_tr + n_v:]
        train_samples += tr
        val_samples   += v
        test_samples  += te
        print(f"  {class_idx_to_name[lbl]}: {len(tr)} train | {len(v)} val | {len(te)} test")

    random.shuffle(train_samples)
    random.shuffle(val_samples)
    random.shuffle(test_samples)

    print(f"\nTOTAL: {len(train_samples)} train | {len(val_samples)} val | {len(test_samples)} test")

    train_loader = DataLoader(RaciDataset(train_samples, train_transform),
                              batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=(device.type == "cuda"))

    val_loader  = DataLoader(RaciDataset(val_samples, eval_transform),
                             batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=2, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(RaciDataset(test_samples, eval_transform),
                             batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=2, pin_memory=(device.type == "cuda"))

    model = build_alexnet_transfer(num_classes=2).to(device)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParametri antrenati: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.1f}%)")

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)

    best_val_acc    = 0.0
    best_model_path = ROOT_DIR / "alexnet_raci_best.pth"

    print(f"\n{'Epoca':>6} | {'Loss Train':>10} | {'Acc Train':>9} | "
          f"{'Loss Val':>8} | {'Acc Val':>7} | {'LR':>8} | {'Timp':>6}")
    print("-" * 70)

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss    = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            tr_loss    += loss.item() * inputs.size(0)
            _, preds    = torch.max(outputs, 1)
            tr_correct += (preds == labels).sum().item()
            tr_total   += labels.size(0)

        train_loss = tr_loss / tr_total
        train_acc  = tr_correct / tr_total

        model.eval()
        vl_loss, vl_correct, vl_total = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss    = criterion(outputs, labels)
                vl_loss    += loss.item() * inputs.size(0)
                _, preds    = torch.max(outputs, 1)
                vl_correct += (preds == labels).sum().item()
                vl_total   += labels.size(0)

        val_loss = vl_loss / vl_total
        val_acc  = vl_correct / vl_total
        elapsed  = time.time() - t0

        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

        current_lr = optimizer.param_groups[0]["lr"]
        gap_warn   = " <- overfitting!" if (train_acc - val_acc) > 0.15 and epoch > 10 else ""

        print(f"{epoch:>6} | {train_loss:>10.4f} | {train_acc:>8.2%} | "
              f"{val_loss:>8.4f} | {val_acc:>6.2%} | {current_lr:>8.5f} | "
              f"{elapsed:>5.1f}s{gap_warn}")

    print(f"\nCel mai bun model: {best_model_path}  (val acc: {best_val_acc:.2%})")

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    ts_correct, ts_total = 0, 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            _, preds    = torch.max(model(inputs), 1)
            ts_correct += (preds == labels).sum().item()
            ts_total   += labels.size(0)

    print(f"Test accuracy: {ts_correct / ts_total:.2%}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_model()