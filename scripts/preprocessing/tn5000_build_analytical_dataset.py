import os, csv, numpy as np, xml.etree.ElementTree as ET
from collections import Counter
from PIL import Image

DATA = r"C:\Users\aupat\Documents\Project's\Thyroide\Thyroide_Project\data"
SAFE = os.path.join(DATA, "TN5000_LeakageSafe")
ANN = os.path.join(DATA, "TN5000_forReview", "Annotations")
OUT = os.path.join(DATA, "TN5000_Analysis")
CSV_OUT = os.path.join(OUT, "thyroid_features.csv")
SUMMARY = os.path.join(OUT, "analytical_dataset_summary.txt")
ERRORS = os.path.join(OUT, "feature_extraction_errors.csv")
os.makedirs(OUT, exist_ok=True)

def entropy(a):
    h = np.bincount(a.ravel(), minlength=256).astype(float)
    p = h[h > 0] / h.sum()
    return float(-(p * np.log2(p)).sum()) if len(p) else 0.0

def edge_density(a):
    a = a.astype(np.float32)
    gx = np.abs(np.diff(a, axis=1)) if a.shape[1] > 1 else np.zeros_like(a)
    gy = np.abs(np.diff(a, axis=0)) if a.shape[0] > 1 else np.zeros_like(a)
    score = np.zeros_like(a)
    if a.shape[1] > 1: score[:, 1:] += gx
    if a.shape[0] > 1: score[1:, :] += gy
    return float(np.mean(score >= 20))

def stats(a, prefix):
    v = a.ravel()
    mean = float(v.mean())
    return {
        prefix+"_mean_intensity": mean,
        prefix+"_std_intensity": float(v.std()),
        prefix+"_min_intensity": int(v.min()),
        prefix+"_max_intensity": int(v.max()),
        prefix+"_median_intensity": float(np.median(v)),
        prefix+"_p10_intensity": float(np.percentile(v,10)),
        prefix+"_p25_intensity": float(np.percentile(v,25)),
        prefix+"_p75_intensity": float(np.percentile(v,75)),
        prefix+"_p90_intensity": float(np.percentile(v,90)),
        prefix+"_entropy": entropy(a),
        prefix+"_contrast": float(v.std()/mean) if mean else 0.0,
        prefix+"_edge_density": edge_density(a),
    }

def annotation(image_id):
    root = ET.parse(os.path.join(ANN, image_id+".xml")).getroot()
    objs = root.findall("object")
    if not objs: raise ValueError("No annotation")
    labels=[]; boxes=[]
    for o in objs:
        raw=o.findtext("name","").strip()
        labels.append({"0":"Benign","1":"Malignant"}.get(raw,raw))
        b=o.find("bndbox")
        if b is None: raise ValueError("Missing bounding box")
        boxes.append(tuple(int(float(b.findtext(k,"0"))) for k in ("xmin","ymin","xmax","ymax")))
    if len(set(labels)) != 1: raise ValueError("Conflicting labels")
    return labels[0], boxes

fields = [
"image_id","split","class","class_numeric","width","height","image_area",
"xmin","ymin","xmax","ymax","nodule_width","nodule_height","nodule_area",
"nodule_area_ratio","nodule_aspect_ratio","nodule_relative_width",
"nodule_relative_height","nodule_center_x","nodule_center_y",
"nodule_center_x_ratio","nodule_center_y_ratio","annotated_object_count",
"image_mean_intensity","image_std_intensity","image_min_intensity",
"image_max_intensity","image_median_intensity","image_p10_intensity",
"image_p25_intensity","image_p75_intensity","image_p90_intensity",
"image_entropy","image_contrast","image_edge_density",
"nodule_mean_intensity","nodule_std_intensity","nodule_min_intensity",
"nodule_max_intensity","nodule_median_intensity","nodule_p10_intensity",
"nodule_p25_intensity","nodule_p75_intensity","nodule_p90_intensity",
"nodule_entropy","nodule_contrast","nodule_edge_density"
]

print("="*70)
print("TN5000 - BUILD ANALYTICAL DATASET")
print("="*70)
print("Input:", SAFE)
print("Output:", CSV_OUT)
print()

records=[]
for split in ("train","val","test"):
    for cls in ("benign","malignant"):
        folder=os.path.join(SAFE,split,cls)
        if not os.path.isdir(folder): raise SystemExit("Missing folder: "+folder)
        for fn in sorted(os.listdir(folder)):
            if fn.lower().endswith((".jpg",".jpeg",".png")):
                records.append((split,cls,os.path.join(folder,fn)))

print("[1] Images found:",len(records))
rows=[]; errors=[]

for n,(split,folder_cls,path) in enumerate(records,1):
    image_id=os.path.splitext(os.path.basename(path))[0]
    try:
        cls, boxes=annotation(image_id)
        expected="Benign" if folder_cls=="benign" else "Malignant"
        if cls != expected: raise ValueError(f"Folder/XML class mismatch: {expected}/{cls}")
        with Image.open(path) as im:
            a=np.asarray(im.convert("L"),dtype=np.uint8)
        h,w=a.shape
        xmin=min(b[0] for b in boxes); ymin=min(b[1] for b in boxes)
        xmax=max(b[2] for b in boxes); ymax=max(b[3] for b in boxes)
        xmin=max(0,min(xmin,w-1)); ymin=max(0,min(ymin,h-1))
        xmax=max(xmin+1,min(xmax,w)); ymax=max(ymin+1,min(ymax,h))
        nw,nh=xmax-xmin,ymax-ymin
        ia=w*h; na=nw*nh
        crop=a[ymin:ymax,xmin:xmax]
        if crop.size==0: raise ValueError("Empty nodule crop")
        row={
            "image_id":image_id,"split":split,"class":cls,
            "class_numeric":1 if cls=="Malignant" else 0,
            "width":w,"height":h,"image_area":ia,
            "xmin":xmin,"ymin":ymin,"xmax":xmax,"ymax":ymax,
            "nodule_width":nw,"nodule_height":nh,"nodule_area":na,
            "nodule_area_ratio":na/ia,
            "nodule_aspect_ratio":nw/nh,
            "nodule_relative_width":nw/w,
            "nodule_relative_height":nh/h,
            "nodule_center_x":(xmin+xmax)/2,
            "nodule_center_y":(ymin+ymax)/2,
            "nodule_center_x_ratio":((xmin+xmax)/2)/w,
            "nodule_center_y_ratio":((ymin+ymax)/2)/h,
            "annotated_object_count":len(boxes)
        }
        row.update(stats(a,"image"))
        row.update(stats(crop,"nodule"))
        rows.append(row)
    except Exception as e:
        errors.append([image_id,split,folder_cls,path,str(e)])
    if n%500==0 or n==len(records):
        print(f"  Processed {n}/{len(records)} | successful={len(rows)} | errors={len(errors)}")

with open(CSV_OUT,"w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

with open(ERRORS,"w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["image_id","split","folder_class","image_path","error"]); w.writerows(errors)

sc=Counter(r["split"] for r in rows)
cc=Counter((r["split"],r["class"]) for r in rows)

with open(SUMMARY,"w",encoding="utf-8") as f:
    f.write("TN5000 ANALYTICAL DATASET SUMMARY\n"+"="*70+"\n\n")
    f.write(f"Input images: {len(records)}\nSuccessful rows: {len(rows)}\nErrors: {len(errors)}\nFeatures: {len(fields)}\n\n")
    for s in ("train","val","test"):
        f.write(f"{s}: {sc[s]} | Benign={cc[(s,'Benign')]} | Malignant={cc[(s,'Malignant')]}\n")
    f.write("\nFeatures include metadata, image dimensions, nodule geometry,\nwhole-image intensity/texture statistics, and nodule-region statistics.\n")
    f.write(f"\nSTATUS: {'PASS' if not errors else 'REVIEW REQUIRED'}\n")

print("\n[2] RESULT")
print("Rows:",len(rows))
print("Features:",len(fields))
print("Errors:",len(errors))
for s in ("train","val","test"):
    print(f"  {s}: {sc[s]} (Benign={cc[(s,'Benign')]}, Malignant={cc[(s,'Malignant')]})")
print("\nCSV:",CSV_OUT)
print("Summary:",SUMMARY)
print("Errors:",ERRORS)
print("\nOriginal TN5000 files were NOT modified.")
