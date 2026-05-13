"""
predict.py - inferenta chela vs background
==========================================
Moduri de rulare:

  # Pe o imagine noua (fara masca, doar vizualizare)
  python predict.py --model best_model.pth --input poza.jpg

  # Pe un folder cu imagini noi
  python predict.py --model best_model.pth --input folder/

  # Pe un split din dataset (val sau test) - compara cu masca reala
  python predict.py --model best_model.pth --split val
  python predict.py --model best_model.pth --split test
"""

import argparse
import numpy as np
from pathlib import Path

import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

parser = argparse.ArgumentParser()
parser.add_argument("--model",     default="best_model.pth")
parser.add_argument("--input",     default=None,
                    help="Imagine sau folder cu imagini noi (fara masca)")
parser.add_argument("--split",     default=None, choices=["val", "test"],
                    help="Ruleaza pe dataset/val sau dataset/test (cu masca reala)")
parser.add_argument("--data_dir",  default="dataset",
                    help="Folderul dataset (folosit cu --split)")
parser.add_argument("--out_dir",   default="predictions")
parser.add_argument("--img_size",  type=int,   default=512)
parser.add_argument("--threshold", type=float, default=0.5)
args = parser.parse_args()

if args.input is None and args.split is None:
    raise ValueError("Trebuie sa dai --input SAU --split val/test")

# ──────────────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

checkpoint = torch.load(args.model, map_location=device, weights_only=False) 
backbone   = checkpoint["backbone"]
print(f"Backbone: {backbone} | Epoca: {checkpoint['epoch']} | Val Dice: {checkpoint['val_dice']:.4f}")

model = smp.Unet(
    encoder_name=backbone,
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation=None,
)
model.load_state_dict(checkpoint["model_state"])
model = model.to(device).eval()

MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

transform = A.Compose([
    A.Resize(args.img_size, args.img_size),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# ──────────────────────────────────────────────────────────────────────────────
def predict_one(img_path):
    original = np.array(Image.open(img_path).convert("RGB"))
    H_orig, W_orig = original.shape[:2]  
    
    tensor   = transform(image=original)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.sigmoid(logits).squeeze().cpu().numpy()
        mask   = (probs > args.threshold)
    
    # Redimensionează probs și mask înapoi la dimensiunea originală
    probs = np.array(Image.fromarray((probs * 255).astype(np.uint8)).resize(
        (W_orig, H_orig), Image.BILINEAR
    )) / 255.0
    
    mask = np.array(Image.fromarray((mask.astype(np.uint8) * 255)).resize(
        (W_orig, H_orig), Image.NEAREST
    )) > 127
    
    return original, probs, mask


def dice_np(pred, gt):
    inter = (pred & gt).sum()
    union = pred.sum() + gt.sum()
    return 2 * inter / union if union > 0 else 1.0


def save_viz_no_gt(img_path, original, probs, mask, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    pct = mask.mean() * 100 
    
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")

    im = axes[1].imshow(probs, cmap="hot", vmin=0, vmax=1)
    axes[1].set_title(f"Probabilitate chela ({pct:.1f}% detectat)") 
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    overlay = original.copy().astype(float)
    green = np.zeros_like(original, dtype=float)
    green[mask] = [0, 220, 80]
    overlay = np.clip(overlay * 0.6 + green * 0.4, 0, 255).astype(np.uint8)
    axes[2].imshow(overlay)
    axes[2].set_title(f"Chela detectata (prag {args.threshold})")
    axes[2].axis("off")

    patches = [
        mpatches.Patch(color=[0, 220/255, 80/255], label="chela"),
        mpatches.Patch(color="black", label="background"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=2)
    plt.suptitle(Path(img_path).name, fontsize=11)
    plt.tight_layout()
    out_path = Path(out_dir) / (Path(img_path).stem + "_pred.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def save_viz_with_gt(img_path, original, probs, pred_mask, gt_mask, dice, out_dir):
    """Vizualizare cu ground truth alaturi - pentru val/test."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Ground truth - verde
    gt_overlay = original.copy().astype(float)
    g = np.zeros_like(original, dtype=float)
    g[gt_mask] = [0, 200, 80]
    gt_overlay = np.clip(gt_overlay * 0.5 + g * 0.5, 0, 255).astype(np.uint8)
    axes[1].imshow(gt_overlay)
    axes[1].set_title("Ground Truth (adnotat)")
    axes[1].axis("off")

    # Predictie - albastru
    pred_overlay = original.copy().astype(float)
    b = np.zeros_like(original, dtype=float)
    b[pred_mask] = [30, 144, 255]
    pred_overlay = np.clip(pred_overlay * 0.5 + b * 0.5, 0, 255).astype(np.uint8)
    axes[2].imshow(pred_overlay)
    axes[2].set_title(f"Predictie (prag {args.threshold})")
    axes[2].axis("off")

    # Comparatie: TP=verde, FP=rosu, FN=galben
    comp = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
    tp = pred_mask & gt_mask # True Positive(verde)
    fp = pred_mask & ~gt_mask # False Positive(rosu)
    fn = ~pred_mask & gt_mask # False Negative(galben)
    comp[tp] = [0, 200, 80]    # verde  = detectat corect
    comp[fp] = [220, 50, 50]   # rosu   = detectat gresit (nu era chela)
    comp[fn] = [255, 220, 0]   # galben = ratat (era chela, nu a detectat)
    axes[3].imshow(comp)
    axes[3].set_title(f"F1 SCORE |(Dice: {dice:.4f})")
    axes[3].axis("off")

    patches = [
        mpatches.Patch(color=[0, 200/255, 80/255],  label="True Positive (corect)"),
        mpatches.Patch(color=[220/255, 50/255, 50/255], label="False Positive (detectat gresit)"),
        mpatches.Patch(color=[1, 220/255, 0],        label="False Negative (ratat)"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05))
    plt.suptitle(Path(img_path).name, fontsize=11)
    plt.tight_layout()
    out_path = Path(out_dir) / (Path(img_path).stem + "_pred.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()

# ──────────────────────────────────────────────────────────────────────────────
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

# -- Mod 1: split val/test cu ground truth --
if args.split:
    img_dir  = Path(args.data_dir) / args.split / "images"
    mask_dir = Path(args.data_dir) / args.split / "masks"

    img_paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    print(f"\nRulez pe split '{args.split}': {len(img_paths)} imagini")

    dice_scores = []
    for img_path in img_paths:
        original, probs, pred_mask = predict_one(img_path)

        # Redimensioneaza pred_mask la dimensiunea originala pt comparatie corecta
        H_orig, W_orig = original.shape[:2]
        pred_resized = np.array(
            Image.fromarray(pred_mask.astype(np.uint8) * 255).resize(
                (W_orig, H_orig), Image.NEAREST
            )
        ) > 127

        # Ground truth
        mask_path = mask_dir / (img_path.stem + ".png")
        gt_mask = np.array(Image.open(mask_path).convert("L")) > 127

        dice = dice_np(pred_resized, gt_mask)
        dice_scores.append(dice)
        print(f"  {img_path.name}: Dice = {dice:.4f}")

        save_viz_with_gt(img_path, original, probs, pred_resized, gt_mask, dice, out_dir)

    print(f"\nDice mediu pe {args.split}: {np.mean(dice_scores):.4f}")
    print(f"Dice minim:  {np.min(dice_scores):.4f}")
    print(f"Dice maxim:  {np.max(dice_scores):.4f}")

# -- Mod 2: imagini noi fara masca -- 
else:
    input_path = Path(args.input)
    img_paths = (
        [input_path] if input_path.is_file()
        else sorted(p for p in input_path.iterdir() if p.suffix.lower() in IMG_EXTS)
    )
    print(f"\nProcesez {len(img_paths)} imagini...")
    for img_path in img_paths:
        original, probs, mask = predict_one(img_path)
        pct = mask.mean() * 100
        print(f"  {img_path.name}: {pct:.1f}% pixeli detectati ca chela")
        save_viz_no_gt(img_path, original, probs, mask, out_dir)

print(f"\nRezultate salvate in: {out_dir}/")