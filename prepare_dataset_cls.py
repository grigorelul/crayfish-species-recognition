"""
prepare_dataset_cls.py - Pregătire dataset pentru CLASIFICARE specii de raci
=============================================================================

CE FACE:
    Parcurge folderul BazaDeDateRaci/ (ORIGINALELE, fara augmentari!)
    si face split stratificat 70/15/15 pe imaginile ORIGINALE.

    IMPORTANT - DE CE NU FOLOSIM BazaDeDateRaci_Augmented DIRECT:
        Daca facem split dupa augmentare, aceeasi imagine originala apare
        in train SI in val/test (ca variante augmentate).
        Rezultat: val=1.0, test=1.0, dar pe o poza noua de pe internet = esec total.
        Asta se numeste "data leakage" si e cauza principala de overfitting aparent.

    Solutia corecta:
        1. Split pe ORIGINALE (BazaDeDateRaci/)
        2. Augmentarea se face DOAR pe train, la runtime in DataLoader

UTILIZARE:
    # Sursa = originalele (nu augmentatele!)
    python prepare_dataset_cls.py --data_root BazaDeDateRaci

    # Dry run - afiseaza statistici fara sa copieze
    python prepare_dataset_cls.py --data_root BazaDeDateRaci --dry_run
"""

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_root", default="BazaDeDateRaci",
    help="Folderul cu ORIGINALELE (default: BazaDeDateRaci). NU cel augmentat!"
)
parser.add_argument("--out_dir",  default="dataset_cls")
parser.add_argument("--train",    type=float, default=0.70)
parser.add_argument("--val",      type=float, default=0.15)
parser.add_argument("--seed",     type=int,   default=42)
parser.add_argument("--dry_run",  action="store_true")
args = parser.parse_args()

random.seed(args.seed)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# ──────────────────────────────────────────────────────────────────────────────
# Colecteaza imaginile ORIGINALE grupate pe specie
# ──────────────────────────────────────────────────────────────────────────────
species_imgs: dict[str, list[Path]] = defaultdict(list)

data_root = Path(args.data_root)
if not data_root.exists():
    raise FileNotFoundError(
        f"Folderul sursa nu exista: {data_root}\n"
        f"Asigura-te ca folosesti ORIGINALELE (BazaDeDateRaci), nu cele augmentate!"
    )

for species_dir in sorted(data_root.iterdir()):
    if not species_dir.is_dir():
        continue
    species_name = species_dir.name

    for img_path in sorted(species_dir.iterdir()):
        if img_path.suffix.lower() in IMG_EXTS:
            species_imgs[species_name].append(img_path)

# ──────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  Sursa: {data_root}/ (imagini ORIGINALE)")
print(f"  Specii gasite: {len(species_imgs)}")
print(f"{'='*60}")
total_imgs = 0
for sp, imgs in species_imgs.items():
    print(f"  {sp}: {len(imgs)} imagini")
    total_imgs += len(imgs)
print(f"{'─'*60}")
print(f"  TOTAL: {total_imgs} imagini originale")
print(f"{'='*60}\n")

if total_imgs == 0:
    raise ValueError(
        f"Nu s-au gasit imagini in {data_root}/\n"
        f"Structura asteptata: {data_root}/NumeSpecie/img.jpg"
    )

# ──────────────────────────────────────────────────────────────────────────────
# Split STRATIFICAT pe ORIGINALE
# ──────────────────────────────────────────────────────────────────────────────
splits: dict[str, dict[str, list[Path]]] = {
    "train": defaultdict(list),
    "val":   defaultdict(list),
    "test":  defaultdict(list),
}

for species_name, imgs in species_imgs.items():
    imgs_copy = list(imgs)
    random.shuffle(imgs_copy)

    n       = len(imgs_copy)
    n_train = max(1, round(n * args.train))
    n_val   = max(1, round(n * args.val))
    n_test  = n - n_train - n_val

    splits["train"][species_name] = imgs_copy[:n_train]
    splits["val"][species_name]   = imgs_copy[n_train : n_train + n_val]
    splits["test"][species_name]  = imgs_copy[n_train + n_val:]

    print(f"  {species_name}: train={n_train}, val={n_val}, test={n_test}")

    if n_test == 0:
        print(f"    [WARN] '{species_name}' nu are imagini in test!")
    if n < 10:
        print(f"    [WARN] Prea putine imagini originale ({n})! "
              f"Considera sa colectezi mai multe date.")

# ──────────────────────────────────────────────────────────────────────────────
if args.dry_run:
    print("\n[DRY RUN] Nu s-a copiat nimic. Sterge --dry_run ca sa copiezi.")
    print("\nNOTA: augmentarile se vor face la runtime in train_cls.py,")
    print("      DOAR pe split-ul de train (nu pe val/test).")
else:
    out_root = Path(args.out_dir)
    if out_root.exists():
        print(f"\n[INFO] Sterg {out_root}/ anterior...")
        shutil.rmtree(out_root)

    for split_name, species_dict in splits.items():
        for species_name, img_paths in species_dict.items():
            dest_dir = out_root / split_name / species_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for img_path in img_paths:
                shutil.copy2(img_path, dest_dir / img_path.name)

        total_in_split = sum(len(v) for v in species_dict.values())
        print(f"  [{split_name}] {total_in_split} imagini copiate (originale)")

    classes_file = out_root / "classes.txt"
    with open(classes_file, "w", encoding="utf-8") as f:
        for sp in sorted(species_imgs.keys()):
            f.write(sp + "\n")

    print(f"\n  Clase salvate in: {classes_file}")
    print(f"\nGata! Dataset in: {out_root}/")
    print("IMPORTANT: augmentarile se aplica automat la runtime in train_cls.py")
    print("Acum ruleaza: python train_cls.py")