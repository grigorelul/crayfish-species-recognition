import json
from pathlib import Path
from PIL import Image

# ── Configurare ──────────────────────────────────────────────────────────────

SRC_DIR       = "BazaDeDateRaci"       # folderul cu pozele originale si JSON-urile LabelMe
DST_DIR       = "BazaDeDateRaciCropped"  # unde se salveaza crop-urile
LABEL         = "rac"                  # label-ul din LabelMe
IMG_SIZE      = 227                    # dimensiunea finala (pixeli)
PADDING       = 20                     # spatiu extra in jurul racului (pixeli)
FORCE         = False                  # True = suprascrie crop-urile existente
PREVIEW       = False                  # True = afiseaza crop-urile fara sa salveze
TRUNC_SPLIT   = True                   # True = separa complet/ si truncated/

CLASS_NAMES = {
    "Austropotamobius bihariensis": "Austropotamobius_bihariensis",
    "Austropotamobius torrentium":  "Austropotamobius_torrentium",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# ── Functii ───────────────────────────────────────────────────────────────────

def is_truncated(img_path: Path) -> bool:
    return "_trunc" in img_path.stem.lower()


def load_bboxes(json_path: Path, label: str):
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  [eroare json] {json_path.name}: {e}")
        return []

    bboxes = []
    for shape in data.get("shapes", []):
        if shape.get("label", "").strip().lower() != label.lower():
            continue
        if shape.get("shape_type") != "rectangle":
            continue
        pts = shape["points"]
        x1 = min(pts[0][0], pts[1][0])
        y1 = min(pts[0][1], pts[1][1])
        x2 = max(pts[0][0], pts[1][0])
        y2 = max(pts[0][1], pts[1][1])
        bboxes.append((x1, y1, x2, y2))

    return bboxes


def crop_image(img: Image.Image, bbox, padding: int, img_size: int) -> Image.Image:
    W, H = img.size
    x1 = max(0, int(bbox[0]) - padding)
    y1 = max(0, int(bbox[1]) - padding)
    x2 = min(W, int(bbox[2]) + padding)
    y2 = min(H, int(bbox[3]) + padding)
    return img.crop((x1, y1, x2, y2)).resize((img_size, img_size), Image.LANCZOS)


def preview_crop(img_orig: Image.Image, img_crop: Image.Image, title: str):
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(title, fontsize=11)
        axes[0].imshow(img_orig);  axes[0].set_title("Original"); axes[0].axis("off")
        axes[1].imshow(img_crop);  axes[1].set_title(f"Crop {img_crop.size[0]}x{img_crop.size[1]}"); axes[1].axis("off")
        plt.tight_layout(); plt.show()
    except ImportError:
        print("  matplotlib nu e instalat, PREVIEW ignorat.")


def get_subfolder(base: Path, truncated: bool) -> Path:
    if not TRUNC_SPLIT:
        return base
    return base / ("truncated" if truncated else "complet")


# ── Main ──────────────────────────────────────────────────────────────────────

root    = Path(__file__).resolve().parent
src_dir = root / SRC_DIR
dst_dir = root / DST_DIR

if not src_dir.exists():
    raise SystemExit(f"[eroare] Folderul sursa nu exista: {src_dir}")

if not PREVIEW:
    dst_dir.mkdir(exist_ok=True)

total_ok = total_sarit = total_fara_json = total_fara_bbox = total_erori = 0

for original_name, folder_name in CLASS_NAMES.items():
    src_cls = src_dir / original_name
    dst_cls = dst_dir / folder_name

    if not src_cls.exists():
        print(f"[lipsa] {src_cls}")
        continue

    if not PREVIEW:
        if TRUNC_SPLIT:
            (dst_cls / "complet").mkdir(parents=True, exist_ok=True)
            (dst_cls / "truncated").mkdir(parents=True, exist_ok=True)
        else:
            dst_cls.mkdir(parents=True, exist_ok=True)

    imagini = sorted(f for f in src_cls.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)
    print(f"\n{original_name} ({len(imagini)} imagini)")

    for img_path in imagini:
        json_path = img_path.with_suffix(".json")

        if not json_path.exists():
            print(f"  [fara json]  {img_path.name}")
            total_fara_json += 1
            continue

        bboxes = load_bboxes(json_path, LABEL)
        if not bboxes:
            print(f"  [fara bbox]  {img_path.name}")
            total_fara_bbox += 1
            continue

        truncated = is_truncated(img_path)
        subfolder = get_subfolder(dst_cls, truncated)

        try:
            img = Image.open(img_path).convert("RGB")

            for idx, bbox in enumerate(bboxes):
                stem     = img_path.stem if len(bboxes) == 1 else f"{img_path.stem}_rac{idx}"
                dst_path = subfolder / (stem + ".jpg")

                if not PREVIEW and dst_path.exists() and not FORCE:
                    total_sarit += 1
                    continue

                cropped = crop_image(img, bbox, PADDING, IMG_SIZE)

                if PREVIEW:
                    label_str = "TRUNC" if truncated else "COMPLET"
                    rac_str   = f" rac{idx}" if len(bboxes) > 1 else ""
                    preview_crop(img, cropped, f"{img_path.name} [{label_str}{rac_str}]")
                else:
                    cropped.save(dst_path, "JPEG", quality=95)

                total_ok += 1

        except Exception as e:
            print(f"  [eroare]     {img_path.name}: {e}")
            total_erori += 1

print(f"\n--- sumar ---")
print(f"  salvate:     {total_ok}")
print(f"  sarite:      {total_sarit}")
print(f"  fara json:   {total_fara_json}")
print(f"  fara bbox:   {total_fara_bbox}")
print(f"  erori:       {total_erori}")