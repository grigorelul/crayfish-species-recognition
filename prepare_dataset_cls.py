"""
prepare_dataset_cls.py - Pregătire dataset pentru CLASIFICARE specii de raci
=============================================================================

CE FACE:
    Parcurge folderul BazaDeDateRaci_Augmented/ (același ca la segmentare),
    dar de data asta nu ne interesează JSON-urile LabelMe - ne interesează
    NUMELE FOLDERULUI (= specia).

    Copiază imaginile în dataset_cls/ cu structura pe care o așteaptă
    PyTorch ImageFolder:
        dataset_cls/
            train/
                Astacus astacus/
                    img1.jpg
                    img2.jpg
            val/
                Astacus astacus/
                    img3.jpg
            test/
                ...

    Split STRATIFICAT: fiecare specie are proporțional 70% train, 15% val, 15% test.
    Stratificat = nu amestecăm toate imaginile și împărțim la întâmplare,
                  ci facem split-ul SEPARAT pentru fiecare specie.
    De ce? Să nu ajungem cu o specie rară complet în train și 0 exemple în val/test.

UTILIZARE:
    python prepare_dataset_cls.py

    # Sursa diferita
    python prepare_dataset_cls.py --data_root BazaDeDateRaci_Augmented

    # Proportii diferite
    python prepare_dataset_cls.py --train 0.80 --val 0.10

    # Afisare statistici fara sa copieze nimic (dry run)
    python prepare_dataset_cls.py --dry_run

DOCUMENTATIE UTILA:
    PyTorch ImageFolder (formatul de folder pe care il generam):
        https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html

    Stratified split (conceptul):
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
"""

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Argumente
# ──────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Pregateste dataset_cls/ pentru clasificare specii"
)
parser.add_argument(
    "--data_root", default="BazaDeDateRaci_Augmented",
    help="Folderul sursa cu structura: specie/img.jpg (default: BazaDeDateRaci_Augmented)"
)
parser.add_argument(
    "--out_dir", default="dataset_cls",
    help="Folderul de output (default: dataset_cls)"
)
parser.add_argument(
    "--train", type=float, default=0.70,
    help="Proportia de train (default: 0.70 = 70%%)"
    # ALTERNATIVA: 0.80 daca ai putine imagini per specie
)
parser.add_argument(
    "--val", type=float, default=0.15,
    help="Proportia de val (default: 0.15 = 15%%)"
)
parser.add_argument(
    "--seed", type=int, default=42,
    help="Seed pentru reproductibilitate (default: 42)"
    # 42 e conventional in ML
)
parser.add_argument(
    "--dry_run", action="store_true",
    help="Afiseaza statistici fara sa copieze fisiere"
)
args = parser.parse_args()

random.seed(args.seed)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# ──────────────────────────────────────────────────────────────────────────────
# Colecteaza imaginile grupate pe specie
# ──────────────────────────────────────────────────────────────────────────────
# defaultdict(list) = un dictionar care creeaza automat o lista goala
# pentru chei noi. Util ca sa nu verificam manual daca cheia exista.
species_imgs: dict[str, list[Path]] = defaultdict(list)

data_root = Path(args.data_root)
if not data_root.exists():
    raise FileNotFoundError(f"Folderul sursa nu exista: {data_root}")

for species_dir in sorted(data_root.iterdir()):
    if not species_dir.is_dir():
        continue
    species_name = species_dir.name  # ex: "Astacus astacus"

    for img_path in sorted(species_dir.iterdir()):
        if img_path.suffix.lower() in IMG_EXTS:
            species_imgs[species_name].append(img_path)

# ──────────────────────────────────────────────────────────────────────────────
# Afisare statistici
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  Specii gasite: {len(species_imgs)}")
print(f"{'='*60}")
total_imgs = 0
for sp, imgs in species_imgs.items():
    print(f"  {sp}: {len(imgs)} imagini")
    total_imgs += len(imgs)
print(f"{'─'*60}")
print(f"  TOTAL: {total_imgs} imagini")
print(f"{'='*60}\n")

# ──────────────────────────────────────────────────────────────────────────────
# Split STRATIFICAT per specie
# ──────────────────────────────────────────────────────────────────────────────
# Structura: splits["train"] = {"Astacus astacus": [img1, img2, ...], ...}
splits: dict[str, dict[str, list[Path]]] = {
    "train": defaultdict(list),
    "val":   defaultdict(list),
    "test":  defaultdict(list),
}

for species_name, imgs in species_imgs.items():
    imgs_copy = list(imgs)
    random.shuffle(imgs_copy)

    n = len(imgs_copy)
    n_train = max(1, round(n * args.train))
    n_val   = max(1, round(n * args.val))
    # Restul merge la test
    n_test  = n - n_train - n_val

    splits["train"][species_name] = imgs_copy[:n_train]
    splits["val"][species_name]   = imgs_copy[n_train : n_train + n_val]
    splits["test"][species_name]  = imgs_copy[n_train + n_val :]

    print(f"  {species_name}:")
    print(f"    train={n_train}, val={n_val}, test={n_test}")

    # Avertizare daca test e gol (prea putine imagini)
    if n_test == 0:
        print(f"    [WARN] Specia '{species_name}' nu are imagini in test!")

# ──────────────────────────────────────────────────────────────────────────────
# Copiere fisiere
# ──────────────────────────────────────────────────────────────────────────────
if args.dry_run:
    print("\n[DRY RUN] Nu s-a copiat nimic. Sterge --dry_run ca sa copiezi.")
else:
    out_root = Path(args.out_dir)

    # Curata output-ul anterior daca exista
    if out_root.exists():
        print(f"\n[INFO] Sterg {out_root}/ anterior...")
        shutil.rmtree(out_root)

    for split_name, species_dict in splits.items():
        for species_name, img_paths in species_dict.items():
            dest_dir = out_root / split_name / species_name
            dest_dir.mkdir(parents=True, exist_ok=True)

            for img_path in img_paths:
                shutil.copy2(img_path, dest_dir / img_path.name)

        # Numara total imagini in split
        total_in_split = sum(len(v) for v in species_dict.values())
        print(f"  [{split_name}] {total_in_split} imagini copiate")

    # Salveaza si lista de clase (utila la inferenta)
    classes_file = out_root / "classes.txt"
    with open(classes_file, "w", encoding="utf-8") as f:
        for sp in sorted(species_imgs.keys()):
            f.write(sp + "\n")
    print(f"\n  Clase salvate in: {classes_file}")

    print(f"\nGata! Dataset clasificare in: {out_root}/")
    print("Acum ruleaza: python train_cls.py")