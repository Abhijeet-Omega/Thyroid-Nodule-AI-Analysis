import os
import csv
import hashlib
import xml.etree.ElementTree as ET
from collections import defaultdict, Counter

# ============================================================
# TN5000 DUPLICATE GROUP ANALYSIS
# Purpose:
#   Investigate the 65 duplicate groups that cross train/val/test.
#   This script DOES NOT modify the original dataset.
# ============================================================

DATASET_ROOT = r"C:\Users\aupat\Documents\Project's\Thyroide\Thyroide_Project\data\TN5000_forReview"
OUTPUT_DIR = os.path.join(
    os.path.dirname(DATASET_ROOT),
    "TN5000_Analysis"
)

IMAGE_DIR = os.path.join(DATASET_ROOT, "JPEGImages")
ANNOTATION_DIR = os.path.join(DATASET_ROOT, "Annotations")
IMAGESETS_DIR = os.path.join(DATASET_ROOT, "ImageSets", "Main")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def md5_file(path, chunk_size=1024 * 1024):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def read_split_map():
    split_map = {}

    for split in ["train", "val", "test"]:
        path = os.path.join(IMAGESETS_DIR, split + ".txt")
        if not os.path.isfile(path):
            continue

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                image_id = line.strip()
                if image_id:
                    split_map[image_id] = split

    return split_map


def parse_annotation(image_id):
    xml_path = os.path.join(ANNOTATION_DIR, image_id + ".xml")

    result = {
        "labels": [],
        "boxes": [],
        "annotation_exists": os.path.isfile(xml_path),
        "annotation_error": ""
    }

    if not os.path.isfile(xml_path):
        return result

    try:
        root = ET.parse(xml_path).getroot()

        for obj in root.findall("object"):
            name = obj.findtext("name", default="").strip()

            bbox = obj.find("bndbox")
            if bbox is not None:
                xmin = int(float(bbox.findtext("xmin", default="0")))
                ymin = int(float(bbox.findtext("ymin", default="0")))
                xmax = int(float(bbox.findtext("xmax", default="0")))
                ymax = int(float(bbox.findtext("ymax", default="0")))
                area = max(0, xmax - xmin) * max(0, ymax - ymin)
                box = (xmin, ymin, xmax, ymax, area)
            else:
                box = None

            result["labels"].append(name)
            result["boxes"].append(box)

    except Exception as e:
        result["annotation_error"] = str(e)

    return result


def get_image_dimensions(image_id):
    # PIL is used only for metadata reading.
    try:
        from PIL import Image
        path = os.path.join(IMAGE_DIR, image_id + ".jpg")
        with Image.open(path) as im:
            return im.width, im.height
    except Exception:
        return "", ""


def annotation_signature(info):
    labels = tuple(info["labels"])
    boxes = tuple(info["boxes"])
    return labels, boxes


print("=" * 70)
print("TN5000 DUPLICATE GROUP ANALYSIS")
print("=" * 70)
print("Dataset:", DATASET_ROOT)
print()

if not os.path.isdir(IMAGE_DIR):
    print("ERROR: JPEGImages folder not found.")
    print("Check DATASET_ROOT at the top of this script.")
    raise SystemExit(1)

split_map = read_split_map()

# ------------------------------------------------------------
# 1. Hash every image
# ------------------------------------------------------------
hash_groups = defaultdict(list)
image_files = [
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

print("[1] HASHING IMAGES")
print("Total image files:", len(image_files))

for i, filename in enumerate(sorted(image_files), start=1):
    image_id = os.path.splitext(filename)[0]
    path = os.path.join(IMAGE_DIR, filename)
    file_hash = md5_file(path)
    hash_groups[file_hash].append(image_id)

    if i % 500 == 0 or i == len(image_files):
        print(f"  Processed {i}/{len(image_files)}")

duplicate_groups = [
    (h, sorted(ids))
    for h, ids in hash_groups.items()
    if len(ids) > 1
]

# ------------------------------------------------------------
# 2. Keep only groups crossing official splits
# ------------------------------------------------------------
cross_split_groups = []

for file_hash, ids in duplicate_groups:
    splits = sorted(set(split_map.get(image_id, "unassigned") for image_id in ids))
    if len([s for s in splits if s in {"train", "val", "test"}]) >= 2:
        cross_split_groups.append((file_hash, ids, splits))

print()
print("[2] DUPLICATE GROUPS")
print("Duplicate groups:", len(duplicate_groups))
print("Cross-split duplicate groups:", len(cross_split_groups))
print()

# ------------------------------------------------------------
# 3. Detailed analysis
# ------------------------------------------------------------
detail_path = os.path.join(
    OUTPUT_DIR, "cross_split_duplicate_group_details.csv"
)

summary_path = os.path.join(
    OUTPUT_DIR, "cross_split_duplicate_group_summary.csv"
)

label_summary_path = os.path.join(
    OUTPUT_DIR, "cross_split_duplicate_label_summary.txt"
)

detail_rows = []
summary_rows = []

affected_by_split = Counter()
affected_by_class = Counter()
groups_by_annotation_status = Counter()
groups_by_label_consistency = Counter()

for group_number, (file_hash, ids, splits) in enumerate(
    sorted(cross_split_groups, key=lambda x: x[1][0]),
    start=1
):
    group_infos = []

    for image_id in ids:
        info = parse_annotation(image_id)
        width, height = get_image_dimensions(image_id)
        split = split_map.get(image_id, "unassigned")

        # Convert numeric labels to readable names.
        readable_labels = []
        for label in info["labels"]:
            if label == "1":
                readable_labels.append("Malignant")
            elif label == "0":
                readable_labels.append("Benign")
            else:
                readable_labels.append(label)

        group_infos.append({
            "image_id": image_id,
            "split": split,
            "width": width,
            "height": height,
            "labels": readable_labels,
            "boxes": info["boxes"],
            "raw_labels": info["labels"],
            "annotation_exists": info["annotation_exists"],
            "annotation_error": info["annotation_error"]
        })

        affected_by_split[split] += 1

        for label in readable_labels:
            affected_by_class[label] += 1

        if info["annotation_error"]:
            groups_by_annotation_status["annotation_error"] += 1
        elif not info["annotation_exists"]:
            groups_by_annotation_status["missing_annotation"] += 1

    # Annotation consistency:
    signatures = []
    for info in group_infos:
        labels = tuple(info["raw_labels"])
        boxes = tuple(info["boxes"])
        signatures.append((labels, boxes))

    annotations_identical = len(set(signatures)) == 1

    # Class consistency only
    label_sets = [tuple(info["raw_labels"]) for info in group_infos]
    labels_consistent = len(set(label_sets)) == 1

    if labels_consistent:
        groups_by_label_consistency["consistent"] += 1
    else:
        groups_by_label_consistency["CONFLICTING"] += 1

    if annotations_identical:
        annotation_status = "IDENTICAL"
    else:
        annotation_status = "DIFFERENT"

    split_string = ",".join(sorted(set(s for s in splits if s != "unassigned")))

    summary_rows.append({
        "duplicate_group": group_number,
        "md5": file_hash,
        "images_in_group": len(ids),
        "splits": split_string,
        "labels_consistent": "YES" if labels_consistent else "NO",
        "annotations_identical": "YES" if annotations_identical else "NO",
        "annotation_status": annotation_status
    })

    # One row per image and per annotation object.
    for info in group_infos:
        labels = info["labels"]
        boxes = info["boxes"]

        if not labels:
            detail_rows.append({
                "duplicate_group": group_number,
                "md5": file_hash,
                "image_id": info["image_id"],
                "split": info["split"],
                "class": "",
                "width": info["width"],
                "height": info["height"],
                "xmin": "",
                "ymin": "",
                "xmax": "",
                "ymax": "",
                "bbox_area": "",
                "annotation_identical_within_group":
                    "YES" if annotations_identical else "NO"
            })
        else:
            for idx, label in enumerate(labels):
                box = boxes[idx] if idx < len(boxes) else None

                if box:
                    xmin, ymin, xmax, ymax, area = box
                else:
                    xmin = ymin = xmax = ymax = area = ""

                detail_rows.append({
                    "duplicate_group": group_number,
                    "md5": file_hash,
                    "image_id": info["image_id"],
                    "split": info["split"],
                    "class": label,
                    "width": info["width"],
                    "height": info["height"],
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                    "bbox_area": area,
                    "annotation_identical_within_group":
                        "YES" if annotations_identical else "NO"
                })


# ------------------------------------------------------------
# 4. Write detailed CSV
# ------------------------------------------------------------
detail_fields = [
    "duplicate_group",
    "md5",
    "image_id",
    "split",
    "class",
    "width",
    "height",
    "xmin",
    "ymin",
    "xmax",
    "ymax",
    "bbox_area",
    "annotation_identical_within_group"
]

with open(detail_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=detail_fields)
    writer.writeheader()
    writer.writerows(detail_rows)


summary_fields = [
    "duplicate_group",
    "md5",
    "images_in_group",
    "splits",
    "labels_consistent",
    "annotations_identical",
    "annotation_status"
]

with open(summary_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=summary_fields)
    writer.writeheader()
    writer.writerows(summary_rows)


# ------------------------------------------------------------
# 5. Human-readable summary
# ------------------------------------------------------------
with open(label_summary_path, "w", encoding="utf-8") as f:
    f.write("TN5000 CROSS-SPLIT DUPLICATE ANALYSIS\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Total images: {len(image_files)}\n")
    f.write(f"Duplicate groups: {len(duplicate_groups)}\n")
    f.write(f"Cross-split duplicate groups: {len(cross_split_groups)}\n\n")

    f.write("AFFECTED IMAGES BY SPLIT\n")
    f.write("-" * 30 + "\n")
    for split in ["train", "val", "test", "unassigned"]:
        f.write(f"{split}: {affected_by_split.get(split, 0)}\n")

    f.write("\nAFFECTED ANNOTATIONS BY CLASS\n")
    f.write("-" * 30 + "\n")
    for label in ["Malignant", "Benign", "0", "1", ""]:
        if affected_by_class.get(label, 0):
            f.write(f"{label}: {affected_by_class[label]}\n")

    f.write("\nANNOTATION CONSISTENCY\n")
    f.write("-" * 30 + "\n")
    f.write(
        f"Groups with identical class + bounding-box annotations: "
        f"{groups_by_label_consistency.get('consistent', 0)}\n"
    )
    f.write(
        f"Groups with conflicting class annotations: "
        f"{groups_by_label_consistency.get('CONFLICTING', 0)}\n"
    )

    f.write("\nANNOTATION FILE STATUS\n")
    f.write("-" * 30 + "\n")
    f.write(
        f"Groups with annotation errors: "
        f"{groups_by_annotation_status.get('annotation_error', 0)}\n"
    )
    f.write(
        f"Groups with missing annotations: "
        f"{groups_by_annotation_status.get('missing_annotation', 0)}\n"
    )

    f.write("\nINTERPRETATION\n")
    f.write("-" * 30 + "\n")
    f.write(
        "This analysis confirms whether exact duplicate images crossing "
        "train/validation/test also have identical labels and bounding boxes.\n"
    )
    f.write(
        "Do NOT delete or move images based only on this report. "
        "Use the results to decide the leakage-safe evaluation strategy.\n"
    )

print()
print("[3] RESULTS")
print("Detailed report:")
print(detail_path)
print()
print("Group summary:")
print(summary_path)
print()
print("Human-readable summary:")
print(label_summary_path)
print()

print("[4] QUICK SUMMARY")
print("Cross-split duplicate groups:", len(cross_split_groups))
print("Affected images by split:")
for split in ["train", "val", "test"]:
    print(f"  {split}: {affected_by_split.get(split, 0)}")

print()
print("Label consistency:")
print("  Consistent groups:", groups_by_label_consistency.get("consistent", 0))
print("  Conflicting groups:", groups_by_label_consistency.get("CONFLICTING", 0))

print()
print("DONE")
print("Original TN5000 files were NOT modified.")
