import os, csv, shutil, hashlib
from collections import Counter

DATASET_ROOT = r"C:\Users\aupat\Documents\Project's\Thyroide\Thyroide_Project\data\TN5000_forReview"
ANALYSIS_DIR = os.path.join(os.path.dirname(DATASET_ROOT), "TN5000_Analysis")
INVENTORY_PATH = os.path.join(ANALYSIS_DIR, "option_a_resulting_split_inventory.csv")
OUTPUT_ROOT = os.path.join(os.path.dirname(DATASET_ROOT), "TN5000_LeakageSafe")
IMAGE_ROOT = os.path.join(DATASET_ROOT, "JPEGImages")

def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""): h.update(chunk)
    return h.hexdigest()

print("="*72)
print("TN5000 OPTION A - CREATE LEAKAGE-SAFE MODELING DATASET")
print("="*72)
print("Original dataset:", DATASET_ROOT)
print("New dataset:", OUTPUT_ROOT)
print("\nThe original TN5000 dataset will NOT be modified.\n")

if not os.path.isdir(DATASET_ROOT): raise SystemExit("ERROR: Original TN5000 dataset folder not found.")
if not os.path.isfile(INVENTORY_PATH): raise SystemExit("ERROR: Option A inventory not found:\n"+INVENTORY_PATH)
if not os.path.isdir(IMAGE_ROOT): raise SystemExit("ERROR: JPEGImages folder not found.")

with open(INVENTORY_PATH, encoding="utf-8") as f:
    rows=list(csv.DictReader(f))
keep_rows=[r for r in rows if r["final_status"]=="KEEP"]
remove_rows=[r for r in rows if r["final_status"]=="REMOVE_FROM_MODELING"]
print("[1] INVENTORY")
print("Inventory rows:",len(rows))
print("Images marked KEEP:",len(keep_rows))
print("Images marked REMOVE:",len(remove_rows))

for split in ("train","val","test"):
    for cls in ("benign","malignant"):
        os.makedirs(os.path.join(OUTPUT_ROOT,split,cls),exist_ok=True)

print("\n[2] COPYING RETAINED IMAGES")
counts=Counter(); class_counts=Counter(); missing=[]; copied=[]
for n,row in enumerate(keep_rows,1):
    image_id,split,cls=row["image_id"],row["original_split"],row["class"]
    folder={"Benign":"benign","Malignant":"malignant"}.get(cls)
    if split not in {"train","val","test"} or folder is None:
        missing.append((image_id,"invalid split/class")); continue
    source=None
    for ext in (".jpg",".jpeg",".JPG",".JPEG",".png",".PNG"):
        p=os.path.join(IMAGE_ROOT,image_id+ext)
        if os.path.isfile(p): source=p; break
    if source is None:
        missing.append((image_id,"image not found")); continue
    dest=os.path.join(OUTPUT_ROOT,split,folder,os.path.basename(source))
    shutil.copy2(source,dest)
    counts[split]+=1; class_counts[(split,folder)]+=1
    copied.append([image_id,split,cls,source,dest,md5_file(dest)])
    if n%500==0 or n==len(keep_rows): print(f"  Copied {n}/{len(keep_rows)}")

expected={("train","benign"):1023,("train","malignant"):2436,("val","benign"):114,("val","malignant"):360,("test","benign"):269,("test","malignant"):731}
ok=True
print("\n[3] VERIFICATION")
for split in ("train","val","test"):
    print("\n"+split.upper())
    for cls in ("benign","malignant"):
        actual=class_counts[(split,cls)]; exp=expected[(split,cls)]
        status="PASS" if actual==exp else "FAIL"
        print(f"  {cls}: {actual} (expected {exp}) -> {status}")
        if actual!=exp: ok=False
    print("  Total:",counts[split])

total=sum(counts.values())
print("\nTOTAL COPIED:",total)
print("EXPECTED TOTAL: 4933")
if total!=4933: ok=False
print("Missing/problematic files:",len(missing))
if missing: ok=False

inv_out=os.path.join(ANALYSIS_DIR,"TN5000_LeakageSafe_final_inventory.csv")
with open(inv_out,"w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["image_id","original_split","class","source","destination","md5"]); w.writerows(copied)

report=os.path.join(ANALYSIS_DIR,"TN5000_LeakageSafe_build_report.txt")
with open(report,"w",encoding="utf-8") as f:
    f.write("TN5000 LEAKAGE-SAFE DATASET BUILD REPORT\n"+"="*72+"\n\n")
    f.write("Official TEST preserved; exact duplicate copies crossing splits removed from TRAIN/VAL according to Option A.\n\n")
    for split in ("train","val","test"):
        f.write(f"{split}: {counts[split]} | benign={class_counts[(split,'benign')]} | malignant={class_counts[(split,'malignant')]}\n")
    f.write(f"\nTotal images: {total}\nExpected total: 4933\nMissing/problematic files: {len(missing)}\nVERIFICATION: {'PASS' if ok else 'FAIL'}\n")
    f.write("\nOriginal TN5000_forReview dataset was not modified.\n")

print("\n[4] FINAL RESULT")
print("PASS: Leakage-safe dataset created successfully." if ok else "FAIL: Verification found problems. Do NOT train yet.")
print("\nDataset:",OUTPUT_ROOT)
print("Final inventory:",inv_out)
print("Build report:",report)
print("\n"+"="*72)
print("TN5000 LEAKAGE-SAFE DATASET READY" if ok else "TN5000 LEAKAGE-SAFE DATASET REQUIRES REVIEW")
print("="*72)
