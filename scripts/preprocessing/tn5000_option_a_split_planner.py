import os, csv, hashlib, xml.etree.ElementTree as ET
from collections import defaultdict, Counter

DATASET_ROOT = r"C:\Users\aupat\Documents\Project's\Thyroide\Thyroide_Project\data\TN5000_forReview"
OUTPUT_DIR = os.path.join(os.path.dirname(DATASET_ROOT), "TN5000_Analysis")
IMAGE_DIR = os.path.join(DATASET_ROOT, "JPEGImages")
ANNOTATION_DIR = os.path.join(DATASET_ROOT, "Annotations")
IMAGESETS_DIR = os.path.join(DATASET_ROOT, "ImageSets", "Main")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def read_split(name):
    path = os.path.join(IMAGESETS_DIR, name + ".txt")
    with open(path, encoding="utf-8") as f:
        return [x.strip() for x in f if x.strip()]

def get_class(image_id):
    path = os.path.join(ANNOTATION_DIR, image_id + ".xml")
    try:
        root = ET.parse(path).getroot()
        labels = []
        for obj in root.findall("object"):
            x = obj.findtext("name", "").strip()
            labels.append({"1": "Malignant", "0": "Benign"}.get(x, x or "UNKNOWN"))
        return "|".join(sorted(set(labels))) if labels else "UNKNOWN"
    except Exception:
        return "ANNOTATION_ERROR"

print("=" * 70)
print("TN5000 OPTION A - LEAKAGE-SAFE SPLIT PLANNER")
print("=" * 70)
print("NO FILES WILL BE MODIFIED.")
print()

splits = {s: read_split(s) for s in ("train", "val", "test")}
split_of = {i: s for s, ids in splits.items() for i in ids}

print("[1] OFFICIAL SPLITS")
for s in ("train", "val", "test"):
    print(f"  {s}: {len(splits[s])}")
print()

print("[2] HASHING IMAGES")
files = sorted(f for f in os.listdir(IMAGE_DIR)
               if f.lower().endswith((".jpg", ".jpeg", ".png")))
groups = defaultdict(list)
for n, fn in enumerate(files, 1):
    image_id = os.path.splitext(fn)[0]
    groups[md5_file(os.path.join(IMAGE_DIR, fn))].append(image_id)
    if n % 500 == 0 or n == len(files):
        print(f"  Processed {n}/{len(files)}")

duplicate_groups = [(h, sorted(ids)) for h, ids in groups.items() if len(ids) > 1]
print()
print("Total images:", len(files))
print("Unique hashes:", len(groups))
print("Duplicate groups:", len(duplicate_groups))
print()

remove = set()
actions = []
group_type = Counter()

for group_no, (h, ids) in enumerate(sorted(duplicate_groups, key=lambda x: x[1][0]), 1):
    present = sorted(set(split_of.get(i, "unassigned") for i in ids))
    official = [s for s in present if s in {"train", "val", "test"}]

    if len(official) <= 1:
        group_type["within_one_split"] += 1
        continue

    if "test" in official:
        group_type["involving_test"] += 1
        reason = "Preserve official TEST copy"
        for image_id in ids:
            s = split_of[image_id]
            action = "KEEP" if s == "test" else "REMOVE"
            if action == "REMOVE":
                remove.add(image_id)
            actions.append([group_no, h, image_id, s, action, reason])
    else:
        group_type["train_val_only"] += 1
        reason = "Preserve TRAIN copy when no TEST copy exists"
        for image_id in ids:
            s = split_of[image_id]
            action = "REMOVE" if s == "val" else "KEEP"
            if action == "REMOVE":
                remove.add(image_id)
            actions.append([group_no, h, image_id, s, action, reason])

classes = {i: get_class(i) for i in split_of}
original = Counter(split_of.values())
result_ids = {s: [i for i in ids if i not in remove] for s, ids in splits.items()}
result = {s: len(ids) for s, ids in result_ids.items()}

orig_cls = {s: Counter(classes[i] for i in ids) for s, ids in splits.items()}
res_cls = {s: Counter(classes[i] for i in ids) for s, ids in result_ids.items()}

remaining = []
for h, ids in duplicate_groups:
    kept = [i for i in ids if i not in remove]
    if len(set(split_of[i] for i in kept)) > 1:
        remaining.append((h, kept))

action_path = os.path.join(OUTPUT_DIR, "option_a_duplicate_removal_plan.csv")
with open(action_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["duplicate_group","md5","image_id","original_split","action","reason"])
    w.writerows(actions)

inventory_path = os.path.join(OUTPUT_DIR, "option_a_resulting_split_inventory.csv")
with open(inventory_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["image_id","original_split","final_status","class"])
    for s, ids in splits.items():
        for i in ids:
            w.writerow([i, s, "REMOVE_FROM_MODELING" if i in remove else "KEEP", classes[i]])

report_path = os.path.join(OUTPUT_DIR, "option_a_leakage_safe_plan.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("TN5000 OPTION A - LEAKAGE-SAFE SPLIT PLAN\n")
    f.write("=" * 70 + "\n\n")
    f.write("Strategy: preserve official TEST; remove duplicate TRAIN/VAL copies.\n")
    f.write("For TRAIN/VAL-only duplicates, preserve TRAIN and remove VAL.\n\n")
    f.write("ORIGINAL COUNTS\n")
    for s in ("train","val","test"):
        f.write(f"{s}: {original[s]} | Benign={orig_cls[s]['Benign']} | Malignant={orig_cls[s]['Malignant']}\n")
    f.write("\nDUPLICATE GROUP HANDLING\n")
    f.write(f"Total duplicate groups: {len(duplicate_groups)}\n")
    f.write(f"Groups involving TEST: {group_type['involving_test']}\n")
    f.write(f"TRAIN/VAL-only groups: {group_type['train_val_only']}\n")
    f.write(f"Groups within one split: {group_type['within_one_split']}\n")
    f.write(f"Images planned for removal: {len(remove)}\n")
    f.write("\nRESULTING COUNTS\n")
    for s in ("train","val","test"):
        f.write(f"{s}: {result[s]} | removed={original[s]-result[s]} | Benign={res_cls[s]['Benign']} | Malignant={res_cls[s]['Malignant']}\n")
    f.write(f"\nCross-split duplicate groups remaining: {len(remaining)}\n")
    f.write("\nNO ORIGINAL TN5000 FILES WERE MODIFIED.\n")

print("[3] OPTION A PLAN")
for s in ("train","val","test"):
    print(f"  {s}: {original[s]} -> {result[s]} (remove {original[s]-result[s]})")
print()
for s in ("train","val","test"):
    print(f"  {s}: Benign={res_cls[s]['Benign']}, Malignant={res_cls[s]['Malignant']}")
print()
print("Images planned for removal:", len(remove))
print("Cross-split duplicate groups remaining:", len(remaining))
print()
print("Reports:")
print(action_path)
print(inventory_path)
print(report_path)
print()
print("DONE - NO ORIGINAL FILES MODIFIED.")
