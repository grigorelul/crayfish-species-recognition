"""
prepare_dataset.py - segmentare BINARĂ: chela vs background
=============================================================
JSON-urile LabelMe extragem doar shape-urile cu label "chela"
Orice altceva devine background (0).

Structură input:
    BazaDeDateRaci_Augmented/
        Astacus astacus/
            img.jpg  img.json ...
        Austropotamobius bihariensis/
            img.jpg  img.json ...

Output:
    dataset/
        train/images/  train/masks/
        val/images/    val/masks/
        test/images/   test/masks/
        classes.txt    <- background\nchela

Mască salvată: 0 = background, 255 = chela (PNG grayscale vizibil)
La citire în Dataset: împarți la 255 -> 0 și 1.
python prepare_dataset.py --data_root BazaDeDateRaci_Augmented

# Schimbi proporțiile (80% train, 10% val, 10% test)
python prepare_dataset.py --data_root BazaDeDateRaci_Augmented --train 0.80 --val 0.10

# Sari peste imaginile fără nicio chelă adnotată
python prepare_dataset.py --data_root BazaDeDateRaci_Augmented --skip_empty

# Schimbi numele clasei (dacă în LabelMe ai scris "claw" în loc de "chela")
python prepare_dataset.py --data_root BazaDeDateRaci_Augmented --chela_label claw

# Folder output diferit
python prepare_dataset.py --data_root BazaDeDateRaci_Augmented --out_dir my_dataset



JSON annotations -> split (train/val/test) -> generate masks (0/1) ->convert to PNG (0/255) -> save dataset structure -> train model


"""
# Pregătire dataset
# python prepare_dataset.py --data_root BazaDeDateRaci_Augmented
import json, shutil, random, argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

# ──────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data_root",   default="BazaDeDateRaci_Augmented")
parser.add_argument("--out_dir",     default="dataset")
parser.add_argument("--chela_label", default="chela",
                    help="Cum ai scris clasa în LabelMe (case-insensitive)")
parser.add_argument("--train", type=float, default=0.70)
parser.add_argument("--val",   type=float, default=0.15)
parser.add_argument("--seed",  type=int,   default=42)
parser.add_argument("--skip_empty", action="store_true",
                    help="Sare imaginile fără nicio adnotare 'chela'")
args = parser.parse_args()

random.seed(args.seed)

CHELA_LABEL = args.chela_label.lower()
IMG_EXTS    = {".jpg", ".jpeg", ".png", ".bmp"}
classes     = ["background", "chela"]

# ──────────────────────────────────────────────────────────────────────────────
# Colectează perechi (img, json)
# ──────────────────────────────────────────────────────────────────────────────
pairs = []
data_root = Path(args.data_root)

for species_dir in sorted(data_root.iterdir()):
    if not species_dir.is_dir():
        continue
    print(f"Specia: {species_dir.name}")
    for img_path in sorted(species_dir.iterdir()):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        json_path = img_path.with_suffix(".json")
        if not json_path.exists():
            print(f"  [WARN] JSON lipsa: {img_path.name}")
            continue
        pairs.append((img_path, json_path))

print(f"\nTotal perechi: {len(pairs)}")


def has_chela(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return any(s.get("label","").strip().lower() == CHELA_LABEL
               for s in data.get("shapes", []))


n_with = sum(1 for _, jp in pairs if has_chela(jp))
print(f"Imagini cu chela adnotata: {n_with}/{len(pairs)}")

if args.skip_empty:
    pairs = [(ip, jp) for ip, jp in pairs if has_chela(jp)]
    print(f"Dupa skip_empty: {len(pairs)} imagini")

# ──────────────────────────────────────────────────────────────────────────────
# Funcție: JSON -> mască binară
# ──────────────────────────────────────────────────────────────────────────────
def make_binary_mask(json_path: Path, W: int, H: int):
    """
    Returnează (mask_uint8, n_chele).
    mask_uint8: shape (H,W), valori 0 sau 1.
    """
    with open(json_path) as f:
        data = json.load(f)

    mask    = np.zeros((H, W), dtype=np.uint8)
    n_chele = 0

    for shape in data.get("shapes", []):
        label      = shape.get("label", "").strip().lower()
        pts        = shape.get("points", [])
        shape_type = shape.get("shape_type", "polygon")

        if label != CHELA_LABEL:
            continue   

        n_chele += 1

        if shape_type in ("polygon", "rectangle") and len(pts) >= 3:
            tmp  = Image.fromarray(mask) # Transform numpy-ul în PIL ca să pot folosi ImageDraw
            draw = ImageDraw.Draw(tmp) # desenez pe tmp, apoi dau convert înapoi la numpy
            draw.polygon([tuple(p) for p in pts], fill=1) # Desenez poligonul pe tmp cu valoarea 1 pentru chela, PIL suporta doar duple si apoi umplu cu 1 interiorul(chela)
            mask = np.array(tmp) # Convertesc înapoi la numpy

    return mask, n_chele

# ──────────────────────────────────────────────────────────────────────────────
# Split 70 / 15 / 15
# ──────────────────────────────────────────────────────────────────────────────
random.shuffle(pairs)
n       = len(pairs)
n_train = max(1, round(n * args.train))
n_val   = max(1, round(n * args.val))

splits = {
    "train": pairs[:n_train],
    "val":   pairs[n_train : n_train + n_val],
    "test":  pairs[n_train + n_val :],
}
for k, v in splits.items():
    print(f"  {k:6s}: {len(v)} imagini")

# ──────────────────────────────────────────────────────────────────────────────
# Generează și salvează
# ──────────────────────────────────────────────────────────────────────────────
out_root    = Path(args.out_dir)
total_chele = 0

for split_name, split_pairs in splits.items():
    img_dir  = out_root / split_name / "images"
    mask_dir = out_root / split_name / "masks"
    img_dir.mkdir(parents=True,  exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    for img_path, json_path in split_pairs:
        shutil.copy2(img_path, img_dir / img_path.name)

        with Image.open(img_path) as im:
            W, H = im.size

        mask_arr, n_chele = make_binary_mask(json_path, W, H)
        total_chele += n_chele

        # Salvăm cu 0/255 ca să fie vizibil PNG-ul
        # (Dataset-ul îl va converti înapoi la 0/1 la citire)
        Image.fromarray(mask_arr * 255, mode="L").save(
            mask_dir / (img_path.stem + ".png")
        )

    print(f"  [{split_name}] salvat")

with open(out_root / "classes.txt", "w") as f:
    f.write("background\nchela\n")

print(f"\nTotal shape-uri chela procesate: {total_chele}")
print("Gata! Acum ruleaza train.py")