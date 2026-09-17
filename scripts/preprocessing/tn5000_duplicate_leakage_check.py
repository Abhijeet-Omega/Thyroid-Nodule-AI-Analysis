from pathlib import Path
from collections import defaultdict
import hashlib
import csv

# Exact TN5000 location used in your project
DATASET_DIR = Path(r"C:\Users\aupat\Documents\Project's\Thyroide\Thyroide_Project\data\TN5000_forReview")

IMAGE_DIR = DATASET_DIR / "JPEGImages"
SETS_DIR = DATASET_DIR / "ImageSets" / "Main"
OUTPUT_DIR = DATASET_DIR.parent / "TN5000_Analysis"

def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def read_split(name):
    path = SETS_DIR / f"{name}.txt"
    if not path.exists():
        return []
    return [Path(x.strip()).stem for x in
            path.read_text(errors="ignore").splitlines() if x.strip()]

print("=" * 70)
print("TN5000 DUPLICATE & DATA-LEAKAGE CHECK")
print("=" * 70)
print("Dataset:", DATASET_DIR)

if not IMAGE_DIR.exists():
    raise SystemExit(f"ERROR: Image folder not found: {IMAGE_DIR}")

images = sorted([
    p for p in IMAGE_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}
])

# ------------------------------------------------------------
# 1. Hash every image
# ------------------------------------------------------------
print("\n[1] CALCULATING IMAGE HASHES")
hash_to_files = defaultdict(list)

for i, path in enumerate(images, 1):
    hash_to_files[md5(path)].append(path.name)
    if i % 500 == 0 or i == len(images):
        print(f"Checked {i}/{len(images)}")

duplicate_groups = [files for files in hash_to_files.values() if len(files) > 1]

print("Total images:", len(images))
print("Unique image hashes:", len(hash_to_files))
print("Duplicate groups:", len(duplicate_groups))
print("Images involved in duplicate groups:",
      sum(len(x) for x in duplicate_groups))

# ------------------------------------------------------------
# 2. Read official splits
# ------------------------------------------------------------
splits = {
    "train": set(read_split("train")),
    "val": set(read_split("val")),
    "test": set(read_split("test")),
}

print("\n[2] OFFICIAL SPLITS")
for name, ids in splits.items():
    print(f"{name}: {len(ids)}")

# ------------------------------------------------------------
# 3. Determine whether each duplicate group crosses splits
# ------------------------------------------------------------
def split_members(filename):
    stem = Path(filename).stem
    found = []
    for split, ids in splits.items():
        if stem in ids:
            found.append(split)
    return found

same_split = []
cross_split = []
not_in_split = []

for group in duplicate_groups:
    memberships = set()
    details = []

    for filename in group:
        member_splits = split_members(filename)
        details.append({
            "filename": filename,
            "splits": ",".join(member_splits)
        })
        memberships.update(member_splits)

    if len(memberships) == 0:
        not_in_split.append((group, details))
    elif len(memberships) == 1:
        same_split.append((group, details))
    else:
        cross_split.append((group, details))

print("\n[3] DUPLICATE GROUP DISTRIBUTION")
print("Duplicate groups entirely within one split:",
      len(same_split))
print("Duplicate groups crossing train/val/test:",
      len(cross_split))
print("Duplicate groups not found in official splits:",
      len(not_in_split))

# ------------------------------------------------------------
# 4. Print cross-split leakage candidates
# ------------------------------------------------------------
print("\n[4] CROSS-SPLIT DUPLICATES")

if not cross_split:
    print("NONE FOUND")
else:
    for i, (group, details) in enumerate(cross_split, 1):
        print(f"\nGroup {i}:")
        for item in details:
            print(f"  {item['filename']} -> {item['splits']}")

# ------------------------------------------------------------
# 5. Save complete reports
# ------------------------------------------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rows = []

for group_id, group in enumerate(duplicate_groups, 1):
    memberships = set()
    for filename in group:
        memberships.update(split_members(filename))

    if len(memberships) == 0:
        category = "not_in_official_split"
    elif len(memberships) == 1:
        category = "same_split"
    else:
        category = "CROSS_SPLIT_LEAKAGE"

    for filename in group:
        rows.append({
            "duplicate_group": group_id,
            "filename": filename,
            "category": category,
            "splits": ",".join(split_members(filename))
        })

report = OUTPUT_DIR / "duplicate_split_report.csv"

with open(report, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["duplicate_group", "filename", "category", "splits"]
    )
    writer.writeheader()
    writer.writerows(rows)

# Summary text
summary = OUTPUT_DIR / "duplicate_leakage_summary.txt"
summary.write_text(
    f"""TN5000 DUPLICATE & DATA-LEAKAGE CHECK

Total images: {len(images)}
Unique image hashes: {len(hash_to_files)}
Duplicate groups: {len(duplicate_groups)}
Images in duplicate groups: {sum(len(x) for x in duplicate_groups)}

Duplicate groups within one split: {len(same_split)}
Duplicate groups crossing train/val/test: {len(cross_split)}
Duplicate groups not in official split files: {len(not_in_split)}

IMPORTANT:
A cross-split duplicate is a potential data-leakage issue.
No files in the original dataset were modified.
""",
    encoding="utf-8"
)

print("\n[5] REPORTS")
print("CSV:", report)
print("Summary:", summary)

print("\n" + "=" * 70)
if cross_split:
    print("RESULT: REVIEW REQUIRED - CROSS-SPLIT DUPLICATES FOUND")
else:
    print("RESULT: NO CROSS-SPLIT DUPLICATE GROUPS FOUND")
print("=" * 70)
print("\nOriginal TN5000 files were NOT modified.")
