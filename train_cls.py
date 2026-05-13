"""
train_cls.py - Clasificare specii de raci folosind encoder-ul din UNet
=======================================================================

CE FACE:
    Ia encoder-ul EfficientNet-B4 din modelul UNet deja antrenat (best_model.pth)
    și adaugă deasupra un clasificator mic (MLP = Multi-Layer Perceptron).

    Encoder-ul e ÎNGHEȚAT la început (nu îi modificăm ponderile) - el știe deja
    să extragă caracteristici vizuale bune din imagini (texturi, forme, culori).
    Antrenăm DOAR clasificatorul.

    Opțional, după câteva epoci, poți "dezgheța" și encoder-ul pentru fine-tuning.

ARHITECTURA:
    Imagine (3×512×512)
        ↓
    EfficientNet-B4 encoder [ÎNGHEȚAT] - 17.6M parametri, nu se antrenează
        ↓
    Global Average Pooling - face media spațială: (B, 1792, H, W) → (B, 1792)
        ↓                    1792 = numărul de canale ale encoder-ului B4
    Linear(1792 → 512) + BatchNorm + ReLU + Dropout(0.4)
        ↓
    Linear(512 → N_SPECII)   ← output: un scor per specie
        ↓
    Softmax (implicit în CrossEntropyLoss) → probabilități


UTILIZARE:
    # Standard
    python train_cls.py

    # Cu unfreeze al encoderului dupa 20 epoci (fine-tuning complet)
    python train_cls.py --unfreeze_epoch 20

    # Fara GPU (mai lent)
    python train_cls.py --amp  # scoate --amp daca nu ai GPU

    # Cu alt numar de epoci sau batch size
    python train_cls.py --epochs 50 --batch_size 16

DOCUMENTATIE:
    Transfer Learning (conceptul):
        https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

    EfficientNet:
        https://arxiv.org/abs/1905.11946
        https://pytorch.org/vision/stable/models/efficientnet.html

    CrossEntropyLoss:
        https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

    AdamW optimizer:
        https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html

    Dropout (regularizare, previne overfitting):
        https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import segmentation_models_pytorch as smp
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# Argumente
# ──────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Antrenare clasificator specii de raci pe features UNet"
)
parser.add_argument(
    "--seg_model", default="best_model.pth",
    help="Modelul UNet antrenat anterior (default: best_model.pth)"
)
parser.add_argument(
    "--data_dir", default="dataset_cls",
    help="Folderul cu structura train/val/test/specie/ (default: dataset_cls)"
)
parser.add_argument(
    "--out_model", default="best_cls_model.pth",
    help="Unde salvam cel mai bun model (default: best_cls_model.pth)"
)
parser.add_argument(
    "--epochs", type=int, default=50,
    help="Numar epoci de antrenare (default: 50)"
    # ALTERNATIVA: 30 daca ai date multe, 100 daca ai date putine
)
parser.add_argument(
    "--batch_size", type=int, default=16,
    help="Imagini per batch (default: 16). Scade la 8 daca ai out-of-memory"
)
parser.add_argument(
    "--lr", type=float, default=1e-3,
    help="Learning rate pentru clasificator (default: 0.001)"
    # ALTERNATIVA: 1e-4 mai conservator, 1e-2 mai agresiv
)
parser.add_argument(
    "--img_size", type=int, default=224,
    help="Dimensiunea imaginii de input (default: 224)"
    # 224 e standard pentru modele ImageNet
    # ALTERNATIVA: 512 (mai lent, mai precis), 128 (rapid, mai putin precis)
)
parser.add_argument(
    "--dropout", type=float, default=0.4,
    help="Dropout rate (default: 0.4). Mai mare = mai multa regularizare"
    # ALTERNATIVA: 0.2 (mai permisiv), 0.6 (mai agresiv anti-overfitting)
)
parser.add_argument(
    "--amp", action="store_true",
    help="Mixed precision training (mai rapid pe GPU Nvidia modern)"
)
parser.add_argument(
    "--patience", type=int, default=10,
    help="Early stopping: epoci fara imbunatatire (default: 10)"
)
parser.add_argument(
    "--unfreeze_epoch", type=int, default=0,
    help="Dupa cate epoci dezghetam si encoder-ul pentru fine-tuning (0=niciodata)"
    # ALTERNATIVA: 15 = primele 15 epoci antrenam doar clasificatorul,
    #              apoi dezghetam tot si antrenam cu LR mai mic
)
args = parser.parse_args()

# ──────────────────────────────────────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

# ──────────────────────────────────────────────────────────────────────────────
# Transformari (augmentari pentru train, doar resize+normalize pentru val/test)
# ──────────────────────────────────────────────────────────────────────────────
# MEAN/STD = valorile ImageNet (folosite si la UNet, consistenta importanta!)
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

# Train: augmentari moderate (mai putine ca la UNet pentru ca nu avem masti)
train_transform = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(30),                  # rotatie aleatorie ±30°
    transforms.ColorJitter(                         # variatie de culoare
        brightness=0.3, contrast=0.3,
        saturation=0.2, hue=0.05
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# Val/Test: FARA augmentari (vrem evaluare consistenta)
val_transform = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# ──────────────────────────────────────────────────────────────────────────────
# Dataset si DataLoader
# ──────────────────────────────────────────────────────────────────────────────
# ImageFolder citeste automat structura:
#   dataset_cls/train/Astacus astacus/*.jpg  → clasa 0
#   dataset_cls/train/Austropotamobius/...   → clasa 1
#   etc.
data_root = Path(args.data_dir)
train_ds = datasets.ImageFolder(str(data_root / "train"), transform=train_transform)
val_ds   = datasets.ImageFolder(str(data_root / "val"),   transform=val_transform)
test_ds  = datasets.ImageFolder(str(data_root / "test"),  transform=val_transform)

# class_to_idx: {"Astacus astacus": 0, "Austropotamobius bihariensis": 1, ...}
class_names = train_ds.classes
n_classes   = len(class_names)
print(f"\nClase ({n_classes}): {class_names}")

# Afisam distributia claselor (important sa fie relativ egala)
print("\nDistributie imagini:")
for split_name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
    counts = [0] * n_classes
    for _, label in ds.samples:
        counts[label] += 1
    print(f"  {split_name}: {dict(zip(class_names, counts))}")

train_loader = DataLoader(
    train_ds, batch_size=args.batch_size,
    shuffle=True, num_workers=0, pin_memory=True
)
val_loader = DataLoader(
    val_ds, batch_size=args.batch_size,
    shuffle=False, num_workers=0, pin_memory=True
)

# ──────────────────────────────────────────────────────────────────────────────
# Incarc encoder-ul din UNet pre-antrenat
# ──────────────────────────────────────────────────────────────────────────────
print(f"\nIncarc encoder din: {args.seg_model}")
checkpoint = torch.load(args.seg_model, map_location=device, weights_only=False)
backbone   = checkpoint["backbone"]  # ex: "efficientnet-b4"
print(f"  Backbone: {backbone}")

# Reconstruiesc UNet-ul complet
unet = smp.Unet(
    encoder_name=backbone,
    encoder_weights=None,   # nu mai incarcam imagenet, incarcam din checkpoint
    in_channels=3,
    classes=1,
    activation=None,
)
unet.load_state_dict(checkpoint["model_state"])

# Extragem DOAR encoder-ul - decoderr-ul si capul de segmentare nu ne trebuie
encoder = unet.encoder
del unet   

# INGHETAM encoder-ul: parametrii nu se vor actualiza la backprop
# (torch.no_grad nu e suficient pentru training - trebuie requires_grad=False)
for param in encoder.parameters():
    param.requires_grad = False

encoder = encoder.to(device)
print(f"  Encoder ingheat: {sum(p.numel() for p in encoder.parameters()):,} parametri")

# ──────────────────────────────────────────────────────────────────────────────
# Clasificator MLP
# ──────────────────────────────────────────────────────────────────────────────
# EfficientNet-B4 scoate features de forma (B, 1792, H, W) la ultimul layer.
# Global Average Pooling face media spatiala → (B, 1792).
# Apoi MLP clasifica.

# Aflam automat dimensiunea output-ului encoderului
with torch.no_grad():
    dummy = torch.zeros(1, 3, args.img_size, args.img_size).to(device)
    # encoder() returneaza o lista de feature maps; ultimul e cel mai important
    feats = encoder(dummy)
    # feats[-1] are shape (1, C, H, W) - vrem C
    enc_out_channels = feats[-1].shape[1]
print(f"  Dimensiune features encoder: {enc_out_channels}")

class SpeciesClassifier(nn.Module):
    """
    Clasificator simplu pe features extrase de encoder.

    Arhitectura:
        GAP → Linear → BN → ReLU → Dropout → Linear → (logits)

    BatchNorm1d: normalizeaza activarile, face antrenarea mai stabila.
    ReLU: activare non-liniara standard.
    Dropout: la antrenare "stinge" aleator neuroni (regularizare anti-overfitting).

    ALTERNATIVE:
        - Fara BatchNorm (mai simplu, uneori suficient)
        - 2 layere intermediare (mai expresiv dar mai greu de antrenat)
        - GeM Pooling in loc de GAP (mai bun pentru retrieval, mai complex)
    """
    def __init__(self, enc_channels: int, n_classes: int, dropout: float):
        super().__init__()

        # Global Average Pooling: (B, C, H, W) → (B, C)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # MLP clasificator
        self.classifier = nn.Sequential(
            nn.Flatten(),                           # (B, C, 1, 1) → (B, C)
            nn.Linear(enc_channels, 512),           # reducere dimensionalitate
            nn.BatchNorm1d(512),                    # stabilizeaza antrenarea
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),                  # regularizare
            nn.Linear(512, n_classes),              # output: 1 scor per clasa
            # NU punem Softmax - e inclus in CrossEntropyLoss
        )
    # Ultimul feature map al encoder-ului are shape (B, enc_channels, H, W)
    def forward(self, features_last):
        """
        features_last: ultimul feature map al encoderului, shape (B, C, H, W)
        """
        x = self.gap(features_last)     # (B, C, 1, 1)
        x = self.classifier(x)          # (B, n_classes)
        return x

classifier = SpeciesClassifier(enc_out_channels, n_classes, args.dropout).to(device)
n_cls_params = sum(p.numel() for p in classifier.parameters())
print(f"  Clasificator: {n_cls_params:,} parametri antrenabili")

# ──────────────────────────────────────────────────────────────────────────────
# Loss, Optimizer, Scheduler
# ──────────────────────────────────────────────────────────────────────────────
# CrossEntropyLoss = standard pentru clasificare multi-clasa
# Include intern Softmax + NegativeLogLikelihood
# https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
criterion = nn.CrossEntropyLoss()

# ALTERNATIVA la CrossEntropyLoss:
#   - LabelSmoothingCrossEntropyLoss(smoothing=0.1) - mai robust la etichete gresite
#   - FocalLoss - buna cand clasele sunt dezechilibrate (o specie cu mult mai putine imagini)

# Antrenam DOAR parametrii clasificatorului (encoder e inghetat)
optimizer = torch.optim.AdamW(
    classifier.parameters(),
    lr=args.lr,
    weight_decay=1e-4   # regularizare L2, previne overfitting
)

# ALTERNATIVA la AdamW:
#   - SGD cu momentum=0.9 (clasic, uneori generalizeaza mai bine)
#   - Adam (fara weight decay)

# ReduceLROnPlateau: injumatateste LR daca val_loss nu scade in 'patience' epoci
# ALTERNATIVA:
#   - CosineAnnealingLR (scade LR in forma de cosinus, popular)
#   - StepLR (scade LR la fiecare N epoci cu factor fix)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=5, factor=0.5
)

scaler = GradScaler("cuda", enabled=args.amp)

# ──────────────────────────────────────────────────────────────────────────────
# Functii helper
# ──────────────────────────────────────────────────────────────────────────────
def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Calculeaza procentul de predictii corecte.
    logits: (B, N_CLASSES) - scoruri brute
    labels: (B,) - indicii claselor reale
    """
    preds = logits.argmax(dim=1)        # clasa cu scorul cel mai mare
    return (preds == labels).float().mean().item()


def run_epoch(loader, train_mode: bool):
    """
    Ruleaza o epoca completa (train sau val).
    Returneaza: (loss_mediu, acuratete_medie)
    """
    if train_mode:
        encoder.eval()           # encoder mereu in eval (e inghetat)
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
                # Extrage features din encoder (fara gradient daca inghetat)
                with torch.no_grad():
                    feats = encoder(imgs)   # lista de feature maps
                last_feat = feats[-1]       # ultimul = cel mai bogat

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
best_val_loss    = float("inf")
patience_counter = 0
unfrozen         = False
prev_lr = args.lr


print(f"\nIncep antrenarea: {args.epochs} epoci, batch={args.batch_size}, lr={args.lr}")
print("─" * 60)


for epoch in range(1, args.epochs + 1):

    # ── Optionally unfreeze encoder (fine-tuning complet) ──
    # Dupa --unfreeze_epoch epoci, dezghetam si encoder-ul si antrenam totul
    # cu un learning rate mult mai mic (ca sa nu stricem ce stie deja)
    if args.unfreeze_epoch > 0 and epoch == args.unfreeze_epoch and not unfrozen:
        print(f"\n[Epoch {epoch}] Dezghet encoder pentru fine-tuning!")
        for param in encoder.parameters():
            param.requires_grad = True
        # Adaugam encoder la optimizer cu LR mic (de 10x mai mic ca clasificatorul)
        optimizer.add_param_group({
            "params": encoder.parameters(),
            "lr": args.lr * 0.01,       # LR foarte mic - encoder stie deja multe
            "weight_decay": 1e-5
        })
        unfrozen = True

    train_loss, train_acc = run_epoch(train_loader, train_mode=True)
    val_loss,   val_acc   = run_epoch(val_loader,   train_mode=False)


    scheduler.step(val_loss)
    
    # Afiseaza LR curent daca s-a schimbat
    current_lr = optimizer.param_groups[0]["lr"]
    if epoch == 1 or current_lr != prev_lr:
        print(f"  [LR] Learning rate: {current_lr:.2e}")
    prev_lr = current_lr

    marker = ""
    if val_loss < best_val_loss:
        best_val_loss    = val_loss
        patience_counter = 0
        torch.save({
            "epoch":        epoch,
            "backbone":     backbone,
            "enc_channels": enc_out_channels,
            "n_classes":    n_classes,
            "class_names":  class_names,
            "cls_state":    classifier.state_dict(),
            "enc_state":    encoder.state_dict(),   # salvam si encoder-ul (pt predict)
            "val_loss":     val_loss,
            "val_acc":      val_acc,
            "img_size":     args.img_size,
            "dropout":      args.dropout,
        }, args.out_model)
        marker = " salvat"
    else:
        patience_counter += 1

    print(
        f"Ep {epoch:3d} | "
        f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
        f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f}"
        f"{marker}"
    )

    if patience_counter >= args.patience:
        print(f"\nEarly stopping la epoca {epoch}.")
        break

print(f"\nBest val loss: {best_val_loss:.4f} → {args.out_model}")
print("Acum ruleaza: python predict_cls.py --input imagine.jpg")