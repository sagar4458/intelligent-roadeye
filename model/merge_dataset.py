"""
Intelligent RoadEye — Dataset Merger + Balancer
Merges both datasets and balances classes using augmentation.

Final output:
  data/merged/crack/    → ~800 images (162 original + augmented)
  data/merged/pothole/  → 800 images (randomly sampled from 2712)
  Ratio                 → 1:1 balanced
"""

import os
import cv2
import shutil
import random
import hashlib
import numpy as np
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent.parent

DATASET1_DIR = BASE_DIR / "data" / "raw" / "Potholes or Cracks on Road Image Dataset"
DATASET2_DIR = BASE_DIR / "data" / "raw" / "potholes, cracks and openmanholes (Road Hazards)"

OUTPUT_DIR   = BASE_DIR / "data" / "merged"
CRACK_DIR    = OUTPUT_DIR / "crack"
POTHOLE_DIR  = OUTPUT_DIR / "pothole"

TARGET_PER_CLASS = 800    # target images per class after balancing
RANDOM_SEED      = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_file_hash(filepath: Path) -> str:
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        h.update(f.read(8192))
    return h.hexdigest()


def copy_images(source_dir: Path, dest_dir: Path, prefix: str, seen_hashes: set) -> list:
    """Copy images and return list of copied destination paths."""
    if not source_dir.exists():
        # Try finding folder with similar name
        parent = source_dir.parent
        if parent.exists():
            for folder in parent.iterdir():
                if folder.is_dir() and "pothole" in folder.name.lower():
                    source_dir = folder
                    break
        if not source_dir.exists():
            print(f"  [SKIP] Not found: {source_dir.name}")
            return []

    images  = list(source_dir.rglob("*.jpg")) + list(source_dir.rglob("*.jpeg"))
    copied  = []

    for i, img_path in enumerate(images):
        try:
            file_hash = get_file_hash(img_path)
            if file_hash in seen_hashes:
                continue
            seen_hashes.add(file_hash)

            new_name  = f"{prefix}_{i:05d}.jpg"
            dest_path = dest_dir / new_name
            shutil.copy2(img_path, dest_path)
            copied.append(dest_path)

        except Exception as e:
            print(f"  [ERROR] {img_path.name}: {e}")

    return copied


# ── Augmentation ──────────────────────────────────────────────────────────────
def augment_image(img: np.ndarray, aug_id: int) -> np.ndarray:
    """Apply one of several augmentation strategies based on aug_id."""
    h, w = img.shape[:2]
    strategy = aug_id % 8

    if strategy == 0:
        # Horizontal flip
        return cv2.flip(img, 1)

    elif strategy == 1:
        # Vertical flip
        return cv2.flip(img, 0)

    elif strategy == 2:
        # Rotate 90
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    elif strategy == 3:
        # Rotate 270
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    elif strategy == 4:
        # Brightness increase
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.3, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    elif strategy == 5:
        # Brightness decrease
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 0.7, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    elif strategy == 6:
        # Random rotation -15 to +15 degrees
        angle = random.uniform(-15, 15)
        M     = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h))

    else:
        # Horizontal flip + brightness
        flipped = cv2.flip(img, 1)
        hsv     = cv2.cvtColor(flipped, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.15, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def augment_to_target(image_paths: list, dest_dir: Path, target: int) -> int:
    """
    Augment images until we reach target count.
    Returns total images in dest_dir after augmentation.
    """
    current = len(image_paths)
    needed  = target - current

    if needed <= 0:
        print(f"  No augmentation needed ({current} >= {target})")
        return current

    print(f"  Augmenting {current} → {target} (need {needed} more)...")

    aug_count = 0
    idx       = 0

    while aug_count < needed:
        src_path = image_paths[idx % len(image_paths)]
        img      = cv2.imread(str(src_path))

        if img is None:
            idx += 1
            continue

        aug_img  = augment_image(img, idx)
        aug_name = f"aug_{aug_count:05d}.jpg"
        aug_path = dest_dir / aug_name

        cv2.imwrite(str(aug_path), aug_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        aug_count += 1
        idx       += 1

        if aug_count % 100 == 0:
            print(f"    {aug_count}/{needed} augmented...")

    return current + aug_count


# ── Main ──────────────────────────────────────────────────────────────────────
def merge():
    print("=" * 60)
    print("  Intelligent RoadEye — Dataset Merger + Balancer")
    print("=" * 60)

    # Clean and recreate output dirs
    if OUTPUT_DIR.exists():
        print(f"\nCleaning existing merged directory...")
        shutil.rmtree(OUTPUT_DIR)

    CRACK_DIR.mkdir(parents=True, exist_ok=True)
    POTHOLE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directories created.")

    crack_hashes   = set()
    pothole_hashes = set()

    # ── CRACK IMAGES ──────────────────────────────────────────────────────────
    print("\n── Step 1: Collecting CRACK images ─────────────────")

    crack_src = DATASET2_DIR / "dataset" / "dataset" / "classes" / "cracks" / "images"
    crack_paths = copy_images(crack_src, CRACK_DIR, "crack_d2", crack_hashes)
    print(f"  Original crack images: {len(crack_paths)}")

    # ── POTHOLE IMAGES ────────────────────────────────────────────────────────
    print("\n── Step 2: Collecting POTHOLE images ────────────────")

    pothole_src1 = DATASET1_DIR / "dataset"
    if pothole_src1.exists():
        for folder in pothole_src1.iterdir():
            if folder.is_dir() and "pothole" in folder.name.lower():
                pothole_src1 = folder
                break

    p1 = copy_images(pothole_src1, POTHOLE_DIR, "pothole_d1", pothole_hashes)
    print(f"  Dataset 1 potholes: {len(p1)}")

    pothole_src2 = DATASET2_DIR / "dataset" / "dataset" / "train" / "images"
    p2 = copy_images(pothole_src2, POTHOLE_DIR, "pothole_d2_train", pothole_hashes)
    print(f"  Dataset 2 train potholes: {len(p2)}")

    pothole_src3 = DATASET2_DIR / "dataset" / "dataset" / "valid" / "images"
    p3 = copy_images(pothole_src3, POTHOLE_DIR, "pothole_d2_valid", pothole_hashes)
    print(f"  Dataset 2 valid potholes: {len(p3)}")

    all_pothole_paths = p1 + p2 + p3
    print(f"  Total pothole images collected: {len(all_pothole_paths)}")

    # ── BALANCE ───────────────────────────────────────────────────────────────
    print(f"\n── Step 3: Balancing classes to {TARGET_PER_CLASS} each ─────")

    # Augment crack images to TARGET_PER_CLASS
    print(f"\n  Augmenting CRACK class:")
    final_crack = augment_to_target(crack_paths, CRACK_DIR, TARGET_PER_CLASS)

    # Randomly remove excess pothole images
    print(f"\n  Balancing POTHOLE class:")
    pothole_files = list(POTHOLE_DIR.glob("*.jpg"))
    current_pothole = len(pothole_files)

    if current_pothole > TARGET_PER_CLASS:
        # Randomly keep TARGET_PER_CLASS images
        random.shuffle(pothole_files)
        to_remove = pothole_files[TARGET_PER_CLASS:]
        for f in to_remove:
            f.unlink()
        print(f"  Reduced pothole: {current_pothole} → {TARGET_PER_CLASS}")
        final_pothole = TARGET_PER_CLASS
    else:
        final_pothole = current_pothole
        print(f"  Pothole count: {final_pothole} (no reduction needed)")

    # ── Final Summary ─────────────────────────────────────────────────────────
    final_crack_count   = len(list(CRACK_DIR.glob("*.jpg")))
    final_pothole_count = len(list(POTHOLE_DIR.glob("*.jpg")))

    print("\n" + "=" * 60)
    print("  MERGE + BALANCE COMPLETE")
    print("=" * 60)
    print(f"  Crack images   : {final_crack_count}")
    print(f"  Pothole images : {final_pothole_count}")
    print(f"  Total          : {final_crack_count + final_pothole_count}")

    ratio = max(final_crack_count, final_pothole_count) / max(min(final_crack_count, final_pothole_count), 1)
    print(f"  Class ratio    : {ratio:.2f}x  ({'✓ Balanced' if ratio < 1.5 else '⚠ Still imbalanced'})")
    print(f"\n  Output: {OUTPUT_DIR}")
    print("  Run model/train.py next.\n")


if __name__ == "__main__":
    merge()