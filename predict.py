import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


IMAGE_PATH = None          
SHOW_GUI   = True          

IMG_SIZE    = 227
CLASS_NAMES = ["Austropotamobius bihariensis", "Austropotamobius torrentium"]
MEAN        = [0.485, 0.456, 0.406]
STD         = [0.229, 0.224, 0.225]
MODEL_FILE  = Path(__file__).resolve().parent / "alexnet_raci_best.pth"


def build_model():
    model = models.alexnet(weights=None)
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, len(CLASS_NAMES))
    return model


def load_model(device):
    if not MODEL_FILE.exists():
        sys.exit(f"Modelul nu a fost gasit: {MODEL_FILE}")

    model = build_model()
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    model.to(device)
    model.eval()
    return model



def select_image():
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Selecteaza o imagine",
            filetypes=[("Imagini", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp")]
        )
        root.destroy()

        if not path:
            sys.exit("Nicio imagine selectata.")
        return Path(path)

    except Exception:
        path = input("Cale imagine: ").strip().strip('"').strip("'")
        if not path:
            sys.exit("Nicio cale introdusa.")
        return Path(path)


def preprocess(image_path):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)



def predict(model, tensor, device):
    tensor = tensor.to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)
    pred_idx = torch.argmax(probs, dim=1).item()
    return pred_idx, probs.squeeze().cpu().tolist()



def show_terminal(image_path, pred_idx, probs):
    print(f"\nImagine: {image_path.name}")
    print(f"Predictie: {CLASS_NAMES[pred_idx]} ({probs[pred_idx]*100:.1f}%)")
    for name, prob in zip(CLASS_NAMES, probs):
        print(f"  {name}: {prob*100:.1f}%")


def show_gui(image_path, pred_idx, probs):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Clasificare Raci de Apa Dulce", fontsize=14, fontweight="bold")

        axes[0].imshow(Image.open(image_path).convert("RGB"))
        axes[0].set_title(image_path.name, fontsize=10)
        axes[0].axis("off")

        colors = ["#4CAF50" if i == pred_idx else "#2196F3" for i in range(len(CLASS_NAMES))]
        bars = axes[1].barh(CLASS_NAMES, [p * 100 for p in probs], color=colors)
        axes[1].set_xlim(0, 100)
        axes[1].set_xlabel("Probabilitate (%)")

        for bar, prob in zip(bars, probs):
            axes[1].text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                         f"{prob*100:.1f}%", va="center", fontsize=10)

        axes[1].legend(handles=[
            mpatches.Patch(color="#4CAF50", label="Clasa prezisa"),
            mpatches.Patch(color="#2196F3", label="Alta clasa"),
        ], loc="lower right")

        plt.tight_layout()
        plt.show()

    except ImportError:
        pass


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(device)

    image_path = Path(IMAGE_PATH) if IMAGE_PATH else select_image()

    if not image_path.exists():
        sys.exit(f"Imaginea nu a fost gasita: {image_path}")

    pred_idx, probs = predict(model, preprocess(image_path), device)

    show_terminal(image_path, pred_idx, probs)

    if SHOW_GUI:
        show_gui(image_path, pred_idx, probs)


if __name__ == "__main__":
    main()