"""
predict_cls.py - Inferenta: clasificare specie de rac
======================================================

CE FACE:
    Incarca modelul de clasificare (best_cls_model.pth) si prezice
    specia de rac dintr-o imagine sau un folder de imagini.

    Afiseaza:
    - Specia prezisa (cu cea mai mare probabilitate)
    - Top-3 specii cu probabilitatile lor
    - Bara de incredere (confidence)
    - Optional: salveaza o imagine cu vizualizarea

MODURI:
    # O singura imagine
    python predict_cls.py --input poza.jpg

    # Un folder intreg
    python predict_cls.py --input folder/

    # Pe split-ul de test (evalueaza acuratetea)
    python predict_cls.py --split test

    # Salveaza vizualizarile
    python predict_cls.py --input poza.jpg --save_viz

    # Top-5 in loc de Top-3
    python predict_cls.py --input poza.jpg --top_k 5

DOCUMENTATIE:
    torch.topk (selecteaza top-k valori):
        https://pytorch.org/docs/stable/generated/torch.topk.html

    torch.nn.functional.softmax:
        https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Clasificare specie de rac dintr-o imagine"
)
parser.add_argument(
    "--cls_model", default="best_cls_model.pth",
    help="Modelul de clasificare (default: best_cls_model.pth)"
)
parser.add_argument(
    "--input", default=None,
    help="Imagine sau folder cu imagini noi"
)
parser.add_argument(
    "--split", default=None, choices=["val", "test"],
    help="Evalueaza pe dataset_cls/val sau test (compara cu specia reala)"
)
parser.add_argument(
    "--data_dir", default="dataset_cls",
    help="Folderul dataset clasificare (folosit cu --split)"
)
parser.add_argument(
    "--out_dir", default="predictions_cls",
    help="Unde salvam vizualizarile (default: predictions_cls)"
)
parser.add_argument(
    "--top_k", type=int, default=3,
    help="Afiseaza top-K specii cu probabilitatile lor (default: 3)"
)
parser.add_argument(
    "--save_viz", action="store_true",
    help="Salveaza vizualizari grafice (implicit: doar afisare in terminal)"
)
args = parser.parse_args()

if args.input is None and args.split is None:
    raise ValueError("Trebuie sa dai --input SAU --split val/test")

# ──────────────────────────────────────────────────────────────────────────────
# Device si incarcare model
# ──────────────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

ckpt = torch.load(args.cls_model, map_location=device, weights_only=False)

backbone     = ckpt["backbone"]
enc_channels = ckpt["enc_channels"]
n_classes    = ckpt["n_classes"]
class_names  = ckpt["class_names"]
img_size     = ckpt["img_size"]
dropout      = ckpt["dropout"]

print(f"Backbone: {backbone}")
print(f"Clase ({n_classes}): {class_names}")
print(f"Val Acc: {ckpt['val_acc']:.4f} | Epoca: {ckpt['epoch']}")

# ──────────────────────────────────────────────────────────────────────────────
# Reconstruim encoder + clasificator
# ──────────────────────────────────────────────────────────────────────────────
# Reconstruim exact aceeasi arhitectura ca la antrenare
unet = smp.Unet(
    encoder_name=backbone,
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation=None,
)
encoder = unet.encoder
encoder.load_state_dict(ckpt["enc_state"])
encoder = encoder.to(device).eval()
del unet

# Clasificatorul (identic cu cel din train_cls.py)
class SpeciesClassifier(nn.Module):
    def __init__(self, enc_channels, n_classes, dropout):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(enc_channels, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, n_classes),
        )

    def forward(self, features_last):
        x = self.gap(features_last)
        return self.classifier(x)

classifier = SpeciesClassifier(enc_channels, n_classes, dropout).to(device)
classifier.load_state_dict(ckpt["cls_state"])
classifier.eval()

# ──────────────────────────────────────────────────────────────────────────────
# Transform (fara augmentari la inferenta)
# ──────────────────────────────────────────────────────────────────────────────
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# ──────────────────────────────────────────────────────────────────────────────
# Functie principala de predictie
# ──────────────────────────────────────────────────────────────────────────────
def predict_image(img_path: Path):
    """
    Ruleaza encoder + clasificator pe o imagine.
    Returneaza:
        probs: numpy array (n_classes,) - probabilitate per specie
        top_k_names: lista cu top-k specii
        top_k_probs: lista cu probabilitatile lor
    """
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)  # (1, 3, H, W)

    with torch.no_grad():
        feats  = encoder(tensor)
        logits = classifier(feats[-1])      # (1, n_classes)
        probs  = F.softmax(logits, dim=1)   # (1, n_classes) → probabilitati sumand la 1
        probs  = probs.squeeze().cpu()      # (n_classes,)

    # Top-k specii (sortate descrescator)
    top_k = min(args.top_k, n_classes)
    top_probs, top_indices = torch.topk(probs, top_k)

    top_k_names = [class_names[i] for i in top_indices.tolist()]
    top_k_probs = top_probs.tolist()

    return probs.numpy(), top_k_names, top_k_probs


def print_result(img_name: str, top_k_names: list, top_k_probs: list,
                 true_label: str = None):
    """
    Afiseaza rezultatul in terminal, cu bara de incredere text.
    """
    pred_name = top_k_names[0]
    pred_prob = top_k_probs[0]

    # Bara de text: [████████░░] 83.4%
    bar_len  = 20
    filled   = int(pred_prob * bar_len)
    bar      = "█" * filled + "░" * (bar_len - filled)

    correct_marker = ""
    if true_label is not None:
        correct_marker = " ✓" if pred_name == true_label else f" ✗ (real: {true_label})"

    print(f"\n  {img_name}")
    print(f"  [{bar}] {pred_prob*100:.1f}%  →  {pred_name}{correct_marker}")

    # Top-k
    for i, (name, prob) in enumerate(zip(top_k_names, top_k_probs)):
        prefix = "  ►" if i == 0 else "   "
        print(f"{prefix}  {prob*100:5.1f}%  {name}")


def save_visualization(img_path: Path, probs: np.ndarray,
                       top_k_names: list, top_k_probs: list,
                       out_dir: Path, true_label: str = None):
    """
    Salveaza o imagine de vizualizare cu:
    - Imaginea originala
    - Bar chart cu probabilitatile per specie
    """
    original = np.array(Image.open(img_path).convert("RGB"))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # -- Stanga: imaginea originala --
    axes[0].imshow(original)
    pred_name = top_k_names[0]
    title_color = "green"
    if true_label and pred_name != true_label:
        title_color = "red"
    axes[0].set_title(
        f"Predictie: {pred_name}\n({top_k_probs[0]*100:.1f}% incredere)",
        color=title_color, fontsize=11, fontweight="bold"
    )
    axes[0].axis("off")

    # -- Dreapta: bar chart probabilitati --
    colors = ["#2ecc71" if n == top_k_names[0] else "#3498db" for n in class_names]
    if true_label:
        colors = []
        for n in class_names:
            if n == top_k_names[0] and n == true_label:
                colors.append("#2ecc71")   # verde = corect
            elif n == top_k_names[0] and n != true_label:
                colors.append("#e74c3c")   # rosu = gresit
            elif n == true_label:
                colors.append("#f39c12")   # portocaliu = clasa reala ratata
            else:
                colors.append("#95a5a6")   # gri = alte clase

    # Nume scurte (primele 15 caractere) pentru grafic
    short_names = [n[:20] + "..." if len(n) > 20 else n for n in class_names]

    bars = axes[1].barh(short_names, probs, color=colors)
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Probabilitate")
    axes[1].set_title("Distributie probabilitati specii")

    # Adauga procentele pe bare
    for bar, prob in zip(bars, probs):
        if prob > 0.02:   # afisam doar daca e mai mare de 2%
            axes[1].text(
                min(prob + 0.01, 0.95), bar.get_y() + bar.get_height() / 2,
                f"{prob*100:.1f}%", va="center", fontsize=9
            )

    # Legenda daca avem ground truth
    if true_label:
        patches = [
            mpatches.Patch(color="#2ecc71", label="Corect"),
            mpatches.Patch(color="#e74c3c", label="Prezis gresit"),
            mpatches.Patch(color="#f39c12", label="Clasa reala"),
            mpatches.Patch(color="#95a5a6", label="Altele"),
        ]
        fig.legend(handles=patches, loc="lower center", ncol=4,
                   bbox_to_anchor=(0.5, -0.08))

    plt.suptitle(img_path.name, fontsize=10)
    plt.tight_layout()

    out_path = out_dir / (img_path.stem + "_cls.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Rulare
# ──────────────────────────────────────────────────────────────────────────────
out_dir = Path(args.out_dir)
if args.save_viz:
    out_dir.mkdir(parents=True, exist_ok=True)

# ── Mod 1: Split val/test cu ground truth ──
if args.split:
    from torchvision import datasets as tvdatasets

    split_dir  = Path(args.data_dir) / args.split
    # Obtinem lista de imagini si labeluri din structura de foldere
    split_ds   = tvdatasets.ImageFolder(str(split_dir))
    img_paths  = [Path(p) for p, _ in split_ds.samples]
    true_labels = [split_ds.classes[l] for _, l in split_ds.samples]

    print(f"\nEvaluez pe split '{args.split}': {len(img_paths)} imagini")
    print("─" * 60)

    correct    = 0
    per_class_correct = {c: 0 for c in class_names}
    per_class_total   = {c: 0 for c in class_names}

    for img_path, true_label in tqdm(
        zip(img_paths, true_labels), total=len(img_paths), desc="Clasificare"
    ):
        probs, top_k_names, top_k_probs = predict_image(img_path)
        pred = top_k_names[0]

        print_result(img_path.name, top_k_names, top_k_probs, true_label)

        if pred == true_label:
            correct += 1
        per_class_total[true_label]   += 1
        per_class_correct[true_label] += int(pred == true_label)

        if args.save_viz:
            save_visualization(img_path, probs, top_k_names, top_k_probs,
                               out_dir, true_label)

    overall_acc = correct / len(img_paths)
    print(f"\n{'='*60}")
    print(f"  Acuratete totala pe {args.split}: {overall_acc*100:.1f}% ({correct}/{len(img_paths)})")
    print(f"{'─'*60}")
    print(f"  Acuratete per specie:")
    for cls in class_names:
        tot = per_class_total[cls]
        cor = per_class_correct[cls]
        acc_cls = cor / tot if tot > 0 else 0.0
        bar = "█" * int(acc_cls * 20) + "░" * (20 - int(acc_cls * 20))
        print(f"    [{bar}] {acc_cls*100:5.1f}%  {cls} ({cor}/{tot})")
    print(f"{'='*60}")

# ── Mod 2: Imagini noi fara ground truth ──
else:
    input_path = Path(args.input)
    img_paths  = (
        [input_path] if input_path.is_file()
        else sorted(p for p in input_path.iterdir()
                    if p.suffix.lower() in IMG_EXTS)
    )
    print(f"\nClasific {len(img_paths)} imagini...")

    for img_path in img_paths:
        probs, top_k_names, top_k_probs = predict_image(img_path)
        print_result(img_path.name, top_k_names, top_k_probs)

        if args.save_viz:
            save_visualization(img_path, probs, top_k_names, top_k_probs, out_dir)

if args.save_viz:
    print(f"\nVizualizari salvate in: {out_dir}/")