import os
from PIL import Image
from collections import Counter

DATA_DIR = r"C:\Users\aupat\Documents\Project's\Thyroide\Thyroide_Project\data\Updated_data"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_images(folder):
    images = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if os.path.splitext(file)[1].lower() in IMAGE_EXTENSIONS:
                images.append(os.path.join(root, file))
    return images


def get_class(path):
    folder = os.path.basename(os.path.dirname(path)).lower()
    if "benign" in folder:
        return "Benign"
    if "malignant" in folder:
        return "Malignant"
    return "Unknown"


def analyze_dataset():
    print("\n" + "=" * 65)
    print("             THYROID DATASET VALIDATION")
    print("=" * 65)

    if not os.path.exists(DATA_DIR):
        print("\nERROR: Dataset folder not found:")
        print(DATA_DIR)
        return

    print("\nDataset location:")
    print(DATA_DIR)

    images = get_images(DATA_DIR)
    print("\nTotal image files:", len(images))

    if len(images) == 0:
        print("\nNo images found.")
        return

    classes = Counter(get_class(path) for path in images)

    print("\n" + "-" * 65)
    print("CLASS DISTRIBUTION")
    print("-" * 65)
    for label, count in classes.items():
        percentage = (count / len(images)) * 100
        print(f"{label:15} : {count:6} ({percentage:.2f}%)")

    formats = Counter(os.path.splitext(path)[1].lower() for path in images)

    print("\n" + "-" * 65)
    print("IMAGE FORMATS")
    print("-" * 65)
    for ext, count in formats.items():
        print(f"{ext.upper():15} : {count}")

    valid_images = 0
    corrupt_images = 0
    dimensions = Counter()

    print("\nChecking image integrity...")

    for i, image_path in enumerate(images, start=1):
        try:
            with Image.open(image_path) as img:
                img.verify()

            with Image.open(image_path) as img:
                dimensions[img.size] += 1

            valid_images += 1

        except Exception:
            corrupt_images += 1
            print("Corrupt image:", image_path)

        if i % 500 == 0:
            print(f"Checked {i}/{len(images)} images...")

    print("\n" + "-" * 65)
    print("IMAGE INTEGRITY")
    print("-" * 65)
    print(f"Valid images     : {valid_images}")
    print(f"Corrupt images   : {corrupt_images}")

    print("\n" + "-" * 65)
    print("IMAGE DIMENSIONS")
    print("-" * 65)
    for (width, height), count in dimensions.most_common(15):
        print(f"{width} x {height:4} : {count}")

    unknown = classes.get("Unknown", 0)

    print("\n" + "-" * 65)
    print("LABEL CHECK")
    print("-" * 65)
    print("Unknown-labelled images:", unknown)

    print("\n" + "=" * 65)
    print("FINAL DATASET SUMMARY")
    print("=" * 65)
    print(f"Total images       : {len(images)}")
    print(f"Valid images       : {valid_images}")
    print(f"Corrupt images     : {corrupt_images}")
    print(f"Benign             : {classes.get('Benign', 0)}")
    print(f"Malignant          : {classes.get('Malignant', 0)}")
    print(f"Unknown            : {unknown}")

    if corrupt_images == 0 and unknown == 0:
        print("\nDataset status: READY FOR ANALYSIS")
    else:
        print("\nDataset status: NEEDS CLEANING")

    print("=" * 65)


if __name__ == "__main__":
    analyze_dataset()
