import json
import copy
import cv2
import numpy as np
from pathlib import Path


# Orice transformare aplicata imaginii trebuie aplicata identic si punctelor
# din JSON, altfel poligonul ajunge decalat fata de obiect.

def flip_points_horizontal(points, img_w):
    return [[img_w - x, y] for x, y in points]


def flip_points_vertical(points, img_h):
    return [[x, img_h - y] for x, y in points]


def letterbox_scale_points(points, orig_w, orig_h, target_size=640):
    # Mutam punctele conform aceluiasi scale si padding aplicat de letterbox_resize()
    scale = target_size / max(orig_h, orig_w)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2

    return [[x * scale + pad_x, y * scale + pad_y] for x, y in points]


# Letterbox: scalare proportionala + bari negre in loc sa tragem de imagine.

def letterbox_resize(image, target_size=640):
    h, w = image.shape[:2]

    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # INTER_AREA: media reala a pixelilor acoperiti.
    # https://gist.github.com/georgeblck/e3e0274d725c858ba98b1c36c14e2835
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Canvas negru pe care lipim imaginea centrat
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    return canvas


# Generam 8 variante per imagine.
# Flipurile modifica si poligonul. Transformarile fotometrice nu misca nimic geometric.
# deepcopy la fiecare varianta -- fara el toate variantele modifica acelasi obiect din memorie.

def augment(image, shapes):
    h, w = image.shape[:2]
    variants = []

    variants.append((image.copy(), copy.deepcopy(shapes), "orig"))

    img_fh = cv2.flip(image, 1)  # 1=stanga-dreapta, 0=sus-jos, -1=ambele
    shapes_fh = copy.deepcopy(shapes)
    for s in shapes_fh:
        s["points"] = flip_points_horizontal(s["points"], w)
    variants.append((img_fh, shapes_fh, "flip_h"))

    img_fv = cv2.flip(image, 0)
    shapes_fv = copy.deepcopy(shapes)
    for s in shapes_fv:
        s["points"] = flip_points_vertical(s["points"], h)
    variants.append((img_fv, shapes_fv, "flip_v"))

    # alpha=contrast, beta=offset luminozitate
    img_dark = cv2.convertScaleAbs(image, alpha=1.0, beta=-30)
    variants.append((img_dark, copy.deepcopy(shapes), "bright_-30"))

    img_bright = cv2.convertScaleAbs(image, alpha=1.0, beta=30)
    variants.append((img_bright, copy.deepcopy(shapes), "bright_30"))

    # kernel (5,5): fereastra de 5x5 pixeli, sigma=0 calculat automat
    img_blur = cv2.GaussianBlur(image, (5, 5), 0)
    variants.append((img_blur, copy.deepcopy(shapes), "blur_gauss"))

    # Adunam in float32 ca uint8 nu suporta valori negative, aducem inapoi la [0, 255]
    noise = np.random.normal(0, 12, image.shape).astype(np.float32)
    img_noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    variants.append((img_noisy, copy.deepcopy(shapes), "noise_gauss"))

    # 0.5% pixeli albi, 0.5% pixeli negri
    img_sp = image.copy()
    num_salt = int(np.ceil(0.01 * image.size * 0.5))
    coords_s = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    img_sp[coords_s[0], coords_s[1], :] = 255
    num_pepper = int(np.ceil(0.01 * image.size * 0.5))
    coords_p = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    img_sp[coords_p[0], coords_p[1], :] = 0
    variants.append((img_sp, copy.deepcopy(shapes), "noise_salt_pepper"))

    return variants


# Construim JSON-ul LabelMe pentru varianta augmentata

def build_json(original_json, shapes_new, img_filename, img_w, img_h):
    data = copy.deepcopy(original_json)
    data["shapes"] = shapes_new
    data["imagePath"] = img_filename
    data["imageWidth"] = img_w
    data["imageHeight"] = img_h
    data["imageData"] = None  
    return data


# Convertim poligoanele LabelMe in format YOLO segmentation.
# Coordonatele sunt normalizate la [0, 1]. Format: class_id x1 y1 x2 y2 ... xN yN
# Nu stiu din txt ce poligon e (daca e chela sau tail sau rac)
def shapes_to_yolo_txt(shapes, class_id, img_size):
    lines = []
    for shape in shapes:
        if shape.get("shape_type") != "polygon":
            continue

        norm = []
        for x, y in shape["points"]:
            norm.extend([
                max(0.0, min(1.0, x / img_size)),
                max(0.0, min(1.0, y / img_size)),
            ])

        coords_str = " ".join(f"{v:.6f}" for v in norm)
        lines.append(f"{class_id} {coords_str}")
    return lines


# Procesam fiecare clasa din dataset si salvam tripletele jpg + json + txt.
# Structura asteptata: un subfolder per clasa, fiecare cu perechi .jpg/.json.
# Clasele primesc id-uri in ordine alfabetica dupa numele folderului.

def process_dataset(input_dir, output_dir, image_size=640):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    class_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    class_names = [d.name for d in class_dirs]
    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "classes.txt", "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(name + "\n")

    print("Classes detected:")
    for name, idx in class_to_id.items():
        print(f"  {idx}: {name}")
    print()

    total_saved = 0
    total_skipped = 0

    for class_dir in class_dirs:
        class_id = class_to_id[class_dir.name]
        out_class_dir = output_dir / class_dir.name
        out_class_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing class '{class_dir.name}' (id={class_id}) ...")

        image_paths = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))

        for img_path in image_paths:
            json_path = img_path.with_suffix(".json")

            if not json_path.exists():
                print(f"  [SKIP] No JSON found for: {img_path.name}")
                total_skipped += 1
                continue

            image = cv2.imread(str(img_path))
            if image is None:
                print(f"  [SKIP] Could not read image: {img_path.name}")
                total_skipped += 1
                continue

            orig_h, orig_w = image.shape[:2]

            with open(json_path, "r", encoding="utf-8") as f:
                labelme_json = json.load(f)

            polygon_shapes = [
                s for s in labelme_json.get("shapes", [])
                if s.get("shape_type") == "polygon"
            ]

            if not polygon_shapes:
                print(f"  [SKIP] No polygon annotations in: {json_path.name}")
                total_skipped += 1
                continue

            variants = augment(image, polygon_shapes)

            for img_variant, shapes_variant, suffix in variants:
                base_name = img_path.stem + "_" + suffix

                shapes_scaled = copy.deepcopy(shapes_variant)
                for s in shapes_scaled:
                    s["points"] = letterbox_scale_points(
                        s["points"], orig_w, orig_h, target_size=image_size
                    )

                img_resized = letterbox_resize(img_variant, target_size=image_size)

                cv2.imwrite(str(out_class_dir / f"{base_name}.jpg"), img_resized)

                json_data = build_json(
                    labelme_json, shapes_scaled,
                    f"{base_name}.jpg", image_size, image_size
                )
                with open(out_class_dir / f"{base_name}.json", "w", encoding="utf-8") as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)

                yolo_lines = shapes_to_yolo_txt(shapes_scaled, class_id, image_size)
                with open(out_class_dir / f"{base_name}.txt", "w", encoding="utf-8") as f:
                    f.write("\n".join(yolo_lines))

                total_saved += 1
                print(f"  [OK] {base_name}.jpg")

        print()

    print(f"Done. {total_saved} triplets saved (jpg + json + txt).")
    if total_skipped > 0:
        print(f"Skipped: {total_skipped} files (missing JSON or no polygon annotations).")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="LabelMe polygon dataset augmentation with coordinate transforms."
    )
    parser.add_argument("--input",  required=True, help="Input dataset folder (e.g. BazaDeDateRaci)")
    parser.add_argument("--output", required=True, help="Output folder (e.g. BazaDeDateRaci_Augmented)")
    parser.add_argument("--size",   type=int, default=640, help="Target image size in pixels (default: 640)")
    args = parser.parse_args()

    process_dataset(args.input, args.output, image_size=args.size)

# python augment_labelme.py --input .\BazaDeDateRaci --output .\BazaDeDateRaci_Augmented