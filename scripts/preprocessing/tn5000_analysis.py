from pathlib import Path
from collections import Counter, defaultdict
import csv, hashlib, json, xml.etree.ElementTree as ET
from PIL import Image

# CHANGE THIS ONLY IF YOUR DATASET IS IN A DIFFERENT LOCATION
DATASET_DIR = Path(r"C:\Users\aupat\Documents\Project's\Thyroide\Thyroide_Project\data\TN5000_forReview")

IMG_DIR = DATASET_DIR / "JPEGImages"
XML_DIR = DATASET_DIR / "Annotations"
SETS_DIR = DATASET_DIR / "ImageSets"
OUT = DATASET_DIR.parent / "TN5000_Analysis"
OUT.mkdir(parents=True, exist_ok=True)

def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()

def parse_xml(p):
    root = ET.parse(p).getroot()
    size = root.find("size")
    w = int(size.findtext("width", "0")) if size is not None else 0
    h = int(size.findtext("height", "0")) if size is not None else 0
    objects = []
    for o in root.findall("object"):
        label = o.findtext("name", "").strip()
        b = o.find("bndbox")
        box = None
        if b is not None:
            try:
                box = tuple(int(float(b.findtext(x, "0"))) for x in
                            ("xmin","ymin","xmax","ymax"))
            except Exception:
                pass
        objects.append((label, box))
    return w, h, objects

def split_names():
    d = SETS_DIR / "Main"
    result = {}
    for p in d.glob("*.txt") if d.exists() else []:
        if p.stem.lower() in {"train","val","test","trainval"}:
            result[p.stem.lower()] = [Path(x.strip()).stem for x in
                                      p.read_text(errors="ignore").splitlines()
                                      if x.strip()]
    return result

print("="*65)
print("TN5000 DATASET VALIDATION")
print("="*65)
print("Dataset:", DATASET_DIR)

for d in (IMG_DIR, XML_DIR, SETS_DIR):
    if not d.exists():
        raise SystemExit(f"\nERROR: Missing folder: {d}")

images = sorted([p for p in IMG_DIR.iterdir()
                 if p.is_file() and p.suffix.lower() in {".jpg",".jpeg"}])
xmls = sorted([p for p in XML_DIR.iterdir()
               if p.is_file() and p.suffix.lower()==".xml"])

image_ids = {p.stem for p in images}
xml_ids = {p.stem for p in xmls}
missing_xml = sorted(image_ids - xml_ids)
missing_img = sorted(xml_ids - image_ids)

print("\n[1] FILE COUNTS")
print("JPEG images:", len(images))
print("XML annotations:", len(xmls))
print("Images without XML:", len(missing_xml))
print("XML without image:", len(missing_img))

print("\n[2] IMAGE VALIDATION")
valid = 0
corrupt = []
dims = Counter()
hashes = defaultdict(list)
image_rows = []

for i,p in enumerate(images,1):
    try:
        with Image.open(p) as im:
            im.verify()
        with Image.open(p) as im:
            w,h = im.size
            mode = im.mode
        valid += 1
        dims[(w,h)] += 1
        hashes[md5(p)].append(p.name)
        image_rows.append({"image_id":p.stem,"filename":p.name,
                           "width":w,"height":h,"mode":mode})
    except Exception as e:
        corrupt.append({"filename":p.name,"error":str(e)})
    if i % 500 == 0 or i == len(images):
        print(f"Checked {i}/{len(images)}")

dupes = [v for v in hashes.values() if len(v)>1]
print("Valid images:", valid)
print("Corrupt images:", len(corrupt))
print("Duplicate groups:", len(dupes))
print("\nMost common dimensions:")
for (w,h),n in dims.most_common(15):
    print(f"  {w} x {h}: {n}")

print("\n[3] XML / LABEL ANALYSIS")
labels = Counter()
bad_xml = []
bad_boxes = []
ann_rows = []
label_by_image = {}

for p in xmls:
    try:
        w,h,objects = parse_xml(p)
        if not objects:
            bad_xml.append({"filename":p.name,"error":"No object"})
        label_by_image.setdefault(p.stem,set())
        for label,box in objects:
            labels[label] += 1
            label_by_image[p.stem].add(label)
            if box is None:
                bad_boxes.append({"filename":p.name,"label":label,
                                  "error":"Missing/invalid bounding box"})
                continue
            xmin,ymin,xmax,ymax = box
            bw,bh = xmax-xmin,ymax-ymin
            if bw <= 0 or bh <= 0:
                bad_boxes.append({"filename":p.name,"label":label,
                                  "error":f"Invalid box {box}"})
            ann_rows.append({"image_id":p.stem,"label":label,
                             "xmin":xmin,"ymin":ymin,"xmax":xmax,"ymax":ymax,
                             "bbox_width":max(0,bw),
                             "bbox_height":max(0,bh),
                             "bbox_area":max(0,bw)*max(0,bh)})
    except Exception as e:
        bad_xml.append({"filename":p.name,"error":str(e)})

print("Invalid XML files:", len(bad_xml))
print("Invalid bounding boxes:", len(bad_boxes))
print("Total annotated objects:", sum(labels.values()))
print("Labels:")
for k,v in labels.items():
    print(f"  {k}: {v}")

print("\n[4] OFFICIAL SPLITS")
splits = split_names()
for k in ("train","val","test","trainval"):
    if k in splits:
        print(f"{k}: {len(splits[k])}")

print("\n[5] SPLIT x CLASS")
for split in ("train","val","test"):
    if split not in splits: continue
    c=Counter()
    for name in splits[split]:
        for label in label_by_image.get(name,set()):
            c[label]+=1
    print(split, dict(c))

areas=[r["bbox_area"] for r in ann_rows if r["bbox_area"]>0]
print("\n[6] BOUNDING BOXES")
if areas:
    print("Minimum area:", min(areas))
    print("Maximum area:", max(areas))
    print("Average area:", round(sum(areas)/len(areas),2))

print("\n[7] REFERENCE CHECK")
expected={"images":5000,"malignant":3572,"benign":1428,"train":3500,"val":500,"test":1000}
print("Reference from TN5000 paper: 5000 images / 3572 malignant / 1428 benign / 3500-500-1000 split")
print("Your actual image count:",len(images))
print("Your actual labels:",dict(labels))
print("Your actual splits:",{k:len(v) for k,v in splits.items()})

def write_csv(name, rows):
    if not rows: return
    with open(OUT/name,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

write_csv("image_inventory.csv",image_rows)
write_csv("annotation_inventory.csv",ann_rows)
write_csv("corrupt_images.csv",corrupt)
write_csv("invalid_xml.csv",bad_xml)
write_csv("invalid_boxes.csv",bad_boxes)
write_csv("duplicate_groups.csv",
          [{"count":len(x),"files":" | ".join(x)} for x in dupes])

summary={
    "images":len(images),"valid_images":valid,"corrupt_images":len(corrupt),
    "xml_annotations":len(xmls),"missing_xml":len(missing_xml),
    "missing_images":len(missing_img),"invalid_xml":len(bad_xml),
    "invalid_boxes":len(bad_boxes),"duplicate_groups":len(dupes),
    "labels":dict(labels),
    "splits":{k:len(v) for k,v in splits.items()},
    "dimensions":{f"{w}x{h}":n for (w,h),n in dims.items()}
}
(OUT/"TN5000_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")

print("\n" + "="*65)
print("DONE")
print("Reports saved to:", OUT)
print("Original TN5000 files were NOT modified.")
print("="*65)
