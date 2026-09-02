"""สร้างเอกสาร README.md อัตโนมัติหลังรันเสร็จ อธิบายว่าแต่ละไฟล์ใน
Config.SAVE_PATH (ตัวเลข/log) และ Config.OUTPUT_PATH (ภาพ) เก็บอะไร
และเอาไปทำอะไรต่อได้บ้าง

dynamic — เช็คว่าไฟล์ไหนมีอยู่จริง ณ ตอนเรียกก่อนเขียนคำอธิบาย
ไม่ list ไฟล์ที่ยังไม่ถูกสร้าง

เรียกจาก:
  scripts/run_dfr.py       → เขียน SAVE_PATH/README.md (ผ่าน src/io_utils.py)
  scripts/visualize_dfr.py → เขียน OUTPUT_PATH/README.md (ฟังก์ชันในไฟล์นี้)

หมายเหตุ: SAVE_PATH/README.md ในโปรเจกต์นี้เขียนโดย
src/io_utils.py::write_save_path_readme() (ตามที่ scripts/run_dfr.py
import ใช้จริง) ไม่ใช่ write_save_path_readme() ในไฟล์นี้ — ไฟล์นี้เก็บไว้
ให้ write_output_path_readme() (OUTPUT_PATH) เท่านั้น เพราะ PatchCore repo
ต้นฉบับมี 2 ฟังก์ชันชื่อซ้ำกันคนละไฟล์แต่ใช้จริงแค่ตัวเดียว (ดู README
หัวข้อ "ความต่างจาก PCB-Anomaly-Baselines-PatchCore")

Generates README.md automatically after a run, documenting every file
in Config.SAVE_PATH (numeric/log) and Config.OUTPUT_PATH (images).

Dynamic — checks which files actually exist at call time.

Called from:
  scripts/run_dfr.py       → writes SAVE_PATH/README.md (via src/io_utils.py)
  scripts/visualize_dfr.py → writes OUTPUT_PATH/README.md (function in this file)

Note: SAVE_PATH/README.md in this project is written by
src/io_utils.py::write_save_path_readme() (the one scripts/run_dfr.py
actually imports) — NOT the write_save_path_readme() in this file. This
file is kept only for write_output_path_readme() (OUTPUT_PATH), since the
original PatchCore repo has two same-named functions in different files
but only actually uses one (see README, "Differences from
PCB-Anomaly-Baselines-PatchCore").
"""
import glob
import os
from datetime import datetime
from typing import List


def _size_str(path: str) -> str:
    if not os.path.exists(path):
        return "N/A"
    size = float(os.path.getsize(path))
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def _glob(base: str, pattern: str) -> List[str]:
    return sorted(glob.glob(os.path.join(base, pattern)))


# ══════════════════════════════════════════════════════════════════════
# OUTPUT_PATH — ภาพ .png
# ══════════════════════════════════════════════════════════════════════

def write_output_path_readme(cfg) -> str:
    p = cfg.OUTPUT_PATH
    os.makedirs(p, exist_ok=True)
    s = []
    method = getattr(cfg, "METHOD_NAME", "DFR")

    s.append(f"""# ภาพผลลัพธ์ใน `{p}` (OUTPUT_PATH) — {method}

สร้างอัตโนมัติโดย `src/output_docs.py` เมื่อ {datetime.now().isoformat(timespec='seconds')}

โฟลเดอร์นี้เก็บ **ไฟล์ภาพ (.png) ทั้งหมด** — ไม่มีไฟล์ตัวเลข
(ตัวเลขอยู่ที่ `{cfg.SAVE_PATH}`)

สร้างโดย `scripts/visualize_dfr.py`

---
""")

    image_docs = [
        ("training_history.png",
         "Train loss (normal only) ต่อ epoch ของ CAE — ไม่มี val curve เพราะ "
         "fit() ตาม BaseAnomalyModel ไม่ได้รับ val_loader (ดู README) | "
         "Per-epoch CAE train loss (normal only) — no val curve since fit() "
         "never receives a val_loader per the BaseAnomalyModel interface"),
        ("eda_class_distribution.png",
         "จำนวนภาพต่อ class (normal/anomaly) แยก val/test — เช็ค class imbalance | "
         "Image count per class split by val/test"),
        ("roc_curves.png",
         "ROC curve (FPR vs TPR) val+test พร้อม AUC | ROC curves for val/test with AUC"),
        ("pr_curves.png",
         "Precision-Recall curve val+test | PR curves for val/test"),
        ("confusion_matrices.png",
         "Confusion matrix val+test ที่ deployment threshold | Confusion matrices at deployment threshold"),
        ("score_distributions.png",
         "Histogram ของ regional-reconstruction-error anomaly score แยก "
         "normal/anomaly พร้อมเส้น threshold | "
         "Reconstruction-error anomaly score histogram split by normal/anomaly with threshold line"),
        ("heatmaps_*.png",
         "ตัวอย่าง 20 ภาพ + regional reconstruction-error heatmap overlay แยก val/test | "
         "20 sample images + reconstruction-error heatmap overlay per split"),
        ("gallery_original_*.png",
         "Grid ภาพ RGB ต้นฉบับ | Grid of original RGB images"),
        ("gallery_processed_*.png",
         "Grid ภาพ + heatmap overlay | Grid of images with heatmap overlay"),
    ]

    for pattern, desc in image_docs:
        base_pat = pattern.split(" / ")[0]
        matches = _glob(p, base_pat) if "*" in base_pat else (
            [os.path.join(p, base_pat)] if os.path.exists(os.path.join(p, base_pat)) else [])
        if not matches:
            continue
        found = "\n".join(
            f"  - `{os.path.basename(m)}` ({_size_str(m)})" for m in matches)
        s.append(f"## `{pattern}`\n**พบไฟล์จริง**:\n{found}\n\n{desc}\n\n---\n\n")

    # gallery_* dynamic (browse_gallery output)
    already = set()
    for pat, _ in image_docs:
        for m in _glob(p, pat.split(" / ")[0]):
            already.add(os.path.basename(m))
    remaining = [m for m in _glob(p, "gallery_*.png")
                 if os.path.basename(m) not in already]
    if remaining:
        found = "\n".join(
            f"  - `{os.path.basename(m)}` ({_size_str(m)})" for m in remaining)
        s.append(
            f"## `gallery_{{split}}_{{label}}_{{pred_label}}.png`\n"
            f"**พบไฟล์จริง**:\n{found}\n\n"
            "Gallery 3 คอลัมน์ (ภาพต้นฉบับ | heatmap | overlay) กรองตาม group "
            "(เช่น gallery_test_defect_normal.png = escape case) | "
            "3-column gallery filtered by group (e.g. escape cases)\n\n---\n\n")

    out = os.path.join(p, "README.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("".join(s))
    return out
