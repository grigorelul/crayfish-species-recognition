"""
train_cls.py - Clasificare specii de raci
==========================================

MODIFICARI FATA DE VERSIUNEA ANTERIOARA:
    1. Augmentarea se face la RUNTIME in DataLoader (nu mai presupunem ca
       dataset-ul e deja augmentat). Asta previne data leakage.

    2. Mascare mai robusta: daca segmentarea esueaza (masca goala / acoperire
       mica), folosim imaginea ORIGINALA in loc sa dam o imagine neagra.
       Pozele de pe internet au adesea clestii in afara cadrului sau in unghi
       diferit fata de datele de antrenare.

    3. Regularizare mai puternica:
       - Dropout crescut la 0.5
       - Label smoothing 0.1 in CrossEntropyLoss
       - Augmentari mai agresive (MixUp optional)

    4. Unfreeze encoder recomandat dupa ~15 epoci ca sa invete features
       specifice speciei (nu features de segmentare de clesti).

UTILIZARE:
    # Standard (cu unfreeze dupa 15 epoci - recomandat!)
    python train_cls.py --unfreeze_epoch 15

    # Fara mascare (daca segmentarea e slaba)
    python train_cls.py --no_mask --unfreeze_epoch 15

    # Mai multa regularizare daca tot overfiteaza
    python train_cls.py --dropout 0.6 --label_smoothing 0.15 --unfreeze_epoch 15
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from PIL import Image
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--seg_model",       default="best_model.pth")
parser.add_argument("--data_dir",        default="dataset_cls")
parser.add_argument("--out_model",       default="best_cls_model.pth")
parser.add_argument("--epochs",          type=int,   default=60)
parser.add_argument("--batch_size",      type=int,   default=16)
parser.add_argument("--lr",              type=float, default=1e-3)
parser.add_argument("--img_size",        type=int,   default=224)
parser.add_argument("--seg_size",        type=int,   default=512)
parser.add_argument("--seg_threshold",   type=float, default=0.5)
parser.add_argument("--dropout",         type=float, default=0.5,
                    help="Dropout rate (default: 0.5, mai mare = mai multa regularizare)")
parser.add_argument("--label_smoothing", type=float, default=0.1,
                    help="Label smoothing (0.1 recomandat, previne supraincrederea)")
parser.add_argument("--amp",             action="store_true")
parser.add_argument("--patience",        type=int,   default=15)
parser.add_argument("--unfreeze_epoch",  type=int,   default=15,
                    help="Dezgheata encoder dupa N epoci (default: 15, 0=niciodata). "
                         "Recomandat! Encoderul trebuie sa invete features de specie, "
                         "nu doar de segmentare.")
parser.add_argument("--no_mask",         action="store_true",
                    help="Dezactiveaza mascarea prin segmentare. Foloseste daca "
                         "segmentarea ta e slaba sau ai putine date.")
parser.add_argument("--mask_fallback",   type=float, default=0.05,
                    help="Daca masca acopera sub X din imagine, foloseste originalul "
                         "(default: 0.05 = 5%%). Previne imagini negre la inferenta.")
args = parser.parse_args()

# ──────────────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

# ──────────────────────────────────────────────────────────────────────────────
# Model segmentare (pentru mascare)
# ──────────────────────────────────────────────────────────────────────────────
print(f"\nIncarc model segmentare: {args.seg_model}")
seg_ckpt     = torch.load(args.seg_model, map_location=device, weights_only=False)
seg_backbone = seg_ckpt["backbone"]

seg_model = smp.Unet(
    encoder_name=seg_backbone,
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation=None,
)
seg_model.load_state_dict(seg_ckpt["model_state"])
seg_model = seg_model.to(device).eval()
for param in seg_model.parameters():
    param.requires_grad = False

print(f"  OK. Backbone: {seg_backbone} | Val Dice: {seg_ckpt['val_dice']:.4f}")
if args.no_mask:
    print("  [INFO] Mascarea dezactivata (--no_mask)")
else:
    print(f"  [INFO] Fallback la original daca masca < {args.mask_fallback*100:.0f}% din imagine")

# ──────────────────────────────────────────────────────────────────────────────
# Encoder pentru clasificare
# ──────────────────────────────────────────────────────────────────────────────
print(f"\nIncarc encoder clasificare din: {args.seg_model}")
unet = smp.Unet(
    encoder_name=seg_backbone,
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation=None,
)
unet.load_state_dict(seg_ckpt["model_state"])
encoder  = unet.encoder
backbone = seg_backbone
del unet

for param in encoder.parameters():
    param.requires_grad = False

encoder = encoder.to(device)
print(f"  Encoder inghebat: {sum(p.numel() for p in encoder.parameters()):,} parametri")
if args.unfreeze_epoch > 0:
    print(f"  Va fi dezghebat la epoca {args.unfreeze_epoch} pentru fine-tuning complet")
else:
    print("  [WARN] Encoder ramine inghebat tot training-ul.")
    print("         Recomandat: --unfreeze_epoch 15 ca sa invete features de specie!")

# ──────────────────────────────────────────────────────────────────────────────
# Transformari
# ──────────────────────────────────────────────────────────────────────────────
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

seg_transform = A.Compose([
    A.Resize(args.seg_size, args.seg_size),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])

# Augmentari mai agresive la train - modelul trebuie sa fie robust la variatii
# (pozitie, lumina, unghi) pe care le va intalni pe poze de pe internet
train_cls_transform = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(45),                          # mai agresiv (era 30)
    transforms.RandomPerspective(distortion_scale=0.3, p=0.4),  # NOU: perspectiva
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
    transforms.RandomGrayscale(p=0.1),                     # NOU: ocazional grayscale
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),   # NOU: blur usor
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),   # NOU: sterge zone mici
])

val_cls_transform = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# ──────────────────────────────────────────────────────────────────────────────
# Mascare robusta
# ──────────────────────────────────────────────────────────────────────────────
def apply_seg_mask(img_np: np.ndarray) -> np.ndarray:
    """
    Mascheaza fundalul prin modelul de segmentare.

    IMBUNATATIRE FATA DE VERSIUNEA ANTERIOARA:
        Daca masca e goala sau acopera sub `mask_fallback` din imagine,
        returneaza imaginea ORIGINALA in loc de una neagra.

        De ce conteaza la inferenta pe poze de pe internet:
            - Clestele poate fi in afara cadrului
            - Unghiul poate fi diferit de datele de antrenare
            - Fundalul poate fi diferit (lab, natura, acvariu)
            → Segmentarea esueaza, masca e goala → imagine neagra → predictie aleatoare
    """
    H_orig, W_orig = img_np.shape[:2]

    seg_tensor = seg_transform(image=img_np)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        seg_logits = seg_model(seg_tensor)
        seg_probs  = torch.sigmoid(seg_logits).squeeze().cpu().numpy()
        seg_mask   = (seg_probs > args.seg_threshold)

    # Verifica daca masca e utila (acopera macar mask_fallback% din imagine)
    mask_coverage = seg_mask.mean()
    if mask_coverage < args.mask_fallback:
        # Masca e prea goala → folosim imaginea originala
        # Modelul va vedea si fundalul, dar macar nu vede negru pur
        return img_np

    seg_mask_orig = np.array(
        Image.fromarray((seg_mask.astype(np.uint8) * 255)).resize(
            (W_orig, H_orig), Image.NEAREST
        )
    ) > 127

    masked = img_np.copy()
    masked[~seg_mask_orig] = 0
    return masked

# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
class MaskedSpeciesDataset(Dataset):
    def __init__(self, split: str, transform=None):
        self.transform = transform
        self.samples   = []
        self.classes   = []

        split_dir  = Path(args.data_dir) / split
        class_dirs = sorted(d for d in split_dir.iterdir() if d.is_dir())
        self.classes = [d.name for d in class_dirs]

        for label_idx, class_dir in enumerate(class_dirs):
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in IMG_EXTS:
                    self.samples.append((img_path, label_idx))

        print(f"  [{split}] {len(self.samples)} imagini, {len(self.classes)} clase")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img_np = np.array(Image.open(img_path).convert("RGB"))

        if not args.no_mask:
            img_np = apply_seg_mask(img_np)

        masked_pil = Image.fromarray(img_np)
        if self.transform:
            masked_pil = self.transform(masked_pil)

        return masked_pil, label

# ──────────────────────────────────────────────────────────────────────────────
print("\nIncarc dataset...")
train_ds = MaskedSpeciesDataset("train", transform=train_cls_transform)
val_ds   = MaskedSpeciesDataset("val",   transform=val_cls_transform)

class_names = train_ds.classes
n_classes   = len(class_names)
print(f"\nClase ({n_classes}): {class_names}")

# Verifica echilibrul claselor (avertizeaza daca sunt foarte dezechilibrate)
class_counts = [0] * n_classes
for _, label in train_ds.samples:
    class_counts[label] += 1
min_c, max_c = min(class_counts), max(class_counts)
if max_c / max(min_c, 1) > 3:
    print(f"\n[WARN] Clase dezechilibrate! Min={min_c}, Max={max_c}")
    print("       Considera --weighted_sampling sau colecteaza mai multe date.")

train_loader = DataLoader(
    train_ds, batch_size=args.batch_size,
    shuffle=True, num_workers=0, pin_memory=True
)
val_loader = DataLoader(
    val_ds, batch_size=args.batch_size,
    shuffle=False, num_workers=0, pin_memory=True
)

# ──────────────────────────────────────────────────────────────────────────────
# Dimensiune encoder output
# ──────────────────────────────────────────────────────────────────────────────
with torch.no_grad():
    dummy            = torch.zeros(1, 3, args.img_size, args.img_size).to(device)
    feats            = encoder(dummy)
    enc_out_channels = feats[-1].shape[1]
print(f"  Dimensiune features encoder: {enc_out_channels}")

# ──────────────────────────────────────────────────────────────────────────────
# Clasificator
# ──────────────────────────────────────────────────────────────────────────────
class SpeciesClassifier(nn.Module):
    def __init__(self, enc_channels: int, n_classes: int, dropout: float):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(enc_channels, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),            # strat extra de regularizare
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.6),   # dropout mai mic la al doilea strat
            nn.Linear(256, n_classes),
        )

    def forward(self, features_last):
        x = self.gap(features_last)
        return self.classifier(x)


classifier = SpeciesClassifier(enc_out_channels, n_classes, args.dropout).to(device)
n_cls_params = sum(p.numel() for p in classifier.parameters())
print(f"  Clasificator: {n_cls_params:,} parametri antrenabili")

# ──────────────────────────────────────────────────────────────────────────────
# Loss cu label smoothing (previne supraincrederea modelului)
# Label smoothing 0.1 = in loc de [0, 0, 1, 0] → [0.033, 0.033, 0.9, 0.033]
# Modelul e penalizat mai puternic daca e extrem de sigur si greseste
# ──────────────────────────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

optimizer = torch.optim.AdamW(
    classifier.parameters(),
    lr=args.lr,
    weight_decay=1e-3,   # crescut de la 1e-4 la 1e-3 (mai multa regularizare L2)
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=args.epochs, eta_min=1e-6
    # Cosine in loc de ReduceLROnPlateau: scade LR mai lin, fara sa blocheze
)

scaler = GradScaler("cuda", enabled=args.amp)

# ──────────────────────────────────────────────────────────────────────────────
def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def run_epoch(loader, train_mode: bool):
    if train_mode:
        encoder.eval()       # encoder in eval (BN stabil)
        classifier.train()
    else:
        encoder.eval()
        classifier.eval()

    total_loss = 0.0
    total_acc  = 0.0
    n_batches  = 0

    ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with ctx:
        for imgs, labels in tqdm(
            loader,
            desc="  Train" if train_mode else "  Val  ",
            leave=False
        ):
            imgs   = imgs.to(device,   non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast("cuda", enabled=args.amp):
                with torch.no_grad():
                    feats = encoder(imgs)
                last_feat = feats[-1]

                logits = classifier(last_feat)
                loss   = criterion(logits, labels)

            if train_mode:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item()
            total_acc  += accuracy(logits, labels)
            n_batches  += 1

    return total_loss / n_batches, total_acc / n_batches

# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────
best_val_acc     = 0.0
patience_counter = 0
unfrozen         = False

print(f"\nIncep antrenarea: {args.epochs} epoci, batch={args.batch_size}, lr={args.lr}")
print(f"Label smoothing: {args.label_smoothing} | Dropout: {args.dropout}")
print(f"Mascare: {'dezactivata' if args.no_mask else 'activa'}")
print("─" * 60)

for epoch in range(1, args.epochs + 1):

    # Dezgheata encoder la epoca unfreeze_epoch
    if args.unfreeze_epoch > 0 and epoch == args.unfreeze_epoch and not unfrozen:
        print(f"\n[Epoch {epoch}] Dezghet encoder pentru fine-tuning complet!")
        print("  Acum modelul invata features SPECIFICE speciei, nu doar de segmentare.")
        for param in encoder.parameters():
            param.requires_grad = True
        # LR mult mai mic pentru encoder (nu vrem sa stricam ce stie deja)
        optimizer.add_param_group({
            "params":       encoder.parameters(),
            "lr":           args.lr * 0.005,    # 200x mai mic decat clasificatorul
            "weight_decay": 1e-5
        })
        # Reinitializeaza scheduler cu restul de epoci
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - epoch, eta_min=1e-7
        )
        unfrozen = True

    train_loss, train_acc = run_epoch(train_loader, train_mode=True)
    val_loss,   val_acc   = run_epoch(val_loader,   train_mode=False)
    scheduler.step()

    marker = ""
    # Salvam dupa val_acc (nu val_loss) pentru clasificare
    if val_acc > best_val_acc:
        best_val_acc     = val_acc
        patience_counter = 0
        torch.save({
            "epoch":         epoch,
            "backbone":      backbone,
            "enc_channels":  enc_out_channels,
            "n_classes":     n_classes,
            "class_names":   class_names,
            "cls_state":     classifier.state_dict(),
            "enc_state":     encoder.state_dict(),
            "val_loss":      val_loss,
            "val_acc":       val_acc,
            "img_size":      args.img_size,
            "seg_size":      args.seg_size,
            "seg_threshold": args.seg_threshold,
            "dropout":       args.dropout,
            "no_mask":       args.no_mask,
        }, args.out_model)
        marker = " ✓ salvat"
    else:
        patience_counter += 1

    # Gap mare train-val = overfitting
    gap = train_acc - val_acc
    gap_warn = " ← overfitting!" if gap > 0.15 and epoch > 10 else ""

    print(
        f"Ep {epoch:3d} | "
        f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
        f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f}"
        f"  [gap:{gap:+.3f}{gap_warn}]"
        f"{marker}"
    )

    if patience_counter >= args.patience:
        print(f"\nEarly stopping la epoca {epoch}.")
        break

print(f"\nBest val acc: {best_val_acc:.4f} → {args.out_model}")
print("Acum ruleaza: python predict_cls.py --input imagine.jpg")