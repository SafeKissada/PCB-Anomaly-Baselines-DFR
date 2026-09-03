"""
โหลด MVTec Anomaly Detection dataset (Bergmann et al. CVPR 2019) ตาม
โครงสร้างโฟลเดอร์ที่แจกจริง:

    {MVTEC_ROOT}/{category}/train/good/*.png
    {MVTEC_ROOT}/{category}/test/good/*.png
    {MVTEC_ROOT}/{category}/test/{defect_type}/*.png            (หลายโฟลเดอร์)
    {MVTEC_ROOT}/{category}/ground_truth/{defect_type}/*_mask.png

ต่างจาก src/data/dataset.py (PCB thesis) ตรงที่ train/test เป็นชุดที่
กำหนดมาตายตัวจาก dataset เอง — "ไม่มีการสุ่ม split" ที่นี่เลย และมี
pixel-level ground truth mask ให้ (PCB thesis มีแค่ label ระดับภาพ)

**ตั้งใจ reuse `AnomalyDataset`/`build_transforms`/`make_loader` จาก
src/data/dataset.py ตรงๆ** (import, ไม่ copy โค้ดซ้ำ) เพราะฟังก์ชันเหล่านี้
ไม่ได้ผูกกับ PCB-specific logic เลย (`AnomalyDataset` รับแค่ DataFrame ที่มี
column "path"/"label", `build_transforms` ใช้แค่ cfg.IMAGE_SIZE/COLOR_MODE)
— แปลว่า preprocessing pipeline (resize, ImageNet normalize) เหมือนกัน
เป๊ะกับที่ PCB baseline อื่นใช้ ไม่มี domain-shift จาก preprocessing ต่างกัน
โดยไม่ตั้งใจ

Loads the MVTec Anomaly Detection dataset per its official folder layout.
Deliberately reuses `AnomalyDataset`/`build_transforms`/`make_loader` from
`src/data/dataset.py` directly (imported, not duplicated) since none of
that code is PCB-specific — it only needs a DataFrame with "path"/"label"
columns and `cfg.IMAGE_SIZE`/`COLOR_MODE`.
"""
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from src.data.dataset import AnomalyDataset, build_transforms, make_loader

logger = logging.getLogger("mvtec_dataset")

_VALID_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def discover_categories(mvtec_root: str, known_categories: Tuple[str, ...]) -> Tuple[str, ...]:
    """คืน tuple ของ category ที่มีอยู่จริงใน mvtec_root (เทียบกับ
    known_categories เป็น allowlist) — ไม่ error ถ้ามีไม่ครบ 15 category
    (เผื่อกรณีดาวน์โหลดมาแค่บางหมวดเพื่อ debug)
    """
    root = Path(mvtec_root)
    found = tuple(
        c for c in known_categories
        if (root / c / "train" / "good").is_dir() and (root / c / "test").is_dir()
    )
    if not found:
        raise FileNotFoundError(
            f"ไม่พบ category ที่ใช้ได้เลยใน {mvtec_root!r} — เช็คว่าแตกไฟล์ "
            f"MVTec AD ถูกต้องหรือยัง (ต้องมี {mvtec_root}/<category>/train/good/ "
            f"เป็นอย่างน้อย 1 หมวด) ที่เช็คหา: {known_categories}"
        )
    missing = set(known_categories) - set(found)
    if missing:
        logger.warning(f"ไม่พบ {len(missing)} category ใน {mvtec_root!r}: "
                       f"{sorted(missing)} — จะข้ามไป รันแค่ {len(found)} ที่เจอ")
    return found


def _list_images(d: Path):
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.suffix.lower() in _VALID_EXT)


def scan_mvtec_category(mvtec_root: str, category: str
                         ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Optional[str]]]:
    """สแกนโฟลเดอร์ 1 category คืน (df_train, df_test, mask_lookup)

    df_train: column "path"/"label" — เฉพาะ train/good/* ทั้งหมด (label="normal")
              ตรงตาม paper: เทรน CAE จากภาพ anomaly-free เท่านั้น

    df_test:  column "path"/"label" — รวมทุกโฟลเดอร์ย่อยของ test/ ทั้ง
              "good" (label="normal") และทุก defect_type (label="anomaly")

    mask_lookup: dict {image_path_str: mask_path_str หรือ None}
              None สำหรับภาพ normal (ไม่มี ground truth mask จริง — คือ
              all-zero mask) และสำหรับภาพ anomaly ที่หา mask ไม่เจอ (เตือน
              ด้วย logger.warning ไม่ throw error ทันที เพราะบาง MVTec
              mirror อาจมีไฟล์ไม่ครบ)
    """
    root = Path(mvtec_root) / category
    if not root.is_dir():
        raise FileNotFoundError(f"ไม่พบ category {category!r} ที่ {root}")

    # ── train: good เท่านั้น (ตาม paper, ไม่มีทางเลือกอื่น) ──────────
    train_good = _list_images(root / "train" / "good")
    if not train_good:
        raise FileNotFoundError(
            f"ไม่พบภาพเลยใน {root/'train'/'good'} — เช็คว่าแตกไฟล์ MVTec AD "
            f"ถูกต้อง (path นี้ต้องมีอย่างน้อย 1 ภาพเสมอสำหรับทุก category)")
    df_train = pd.DataFrame({
        "path": [str(p) for p in train_good],
        "label": ["normal"] * len(train_good),
    })

    # ── test: ทุกโฟลเดอร์ย่อย (good + defect type ต่างๆ) ──────────────
    test_dir = root / "test"
    if not test_dir.is_dir():
        raise FileNotFoundError(f"ไม่พบ {test_dir}")

    test_rows = []
    mask_lookup: Dict[str, Optional[str]] = {}
    n_masks_found, n_masks_missing = 0, 0

    for subdir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
        defect_type = subdir.name
        label = "normal" if defect_type == "good" else "anomaly"
        for img_path in _list_images(subdir):
            path_str = str(img_path)
            test_rows.append({"path": path_str, "label": label})

            if label == "normal":
                mask_lookup[path_str] = None  # normal -> all-zero mask (ไม่ต้องมีไฟล์จริง)
                continue

            # MVTec convention: ground_truth/{defect_type}/{stem}_mask.{ext}
            gt_dir = root / "ground_truth" / defect_type
            mask_path = None
            for ext in _VALID_EXT:
                candidate = gt_dir / f"{img_path.stem}_mask{ext}"
                if candidate.exists():
                    mask_path = candidate
                    break
            if mask_path is not None:
                mask_lookup[path_str] = str(mask_path)
                n_masks_found += 1
            else:
                mask_lookup[path_str] = None
                n_masks_missing += 1

    if n_masks_missing > 0:
        logger.warning(
            f"[{category}] หา ground-truth mask ไม่เจอ {n_masks_missing} ไฟล์ "
            f"(จากทั้งหมด {n_masks_found + n_masks_missing} ภาพ anomaly) — "
            f"ภาพเหล่านี้จะถูกนับเป็น all-zero mask (=ไม่มี defect เลย) ซึ่ง "
            f"ผิด และจะทำให้ pixel-ROC-AUC/PRO-AUC เพี้ยน เช็คว่าแตกไฟล์ "
            f"MVTec AD ครบหรือยัง")

    df_test = pd.DataFrame(test_rows)
    logger.info(f"[{category}] train(normal)={len(df_train)}  "
               f"test={len(df_test)} ({(df_test['label']=='normal').sum()} normal, "
               f"{(df_test['label']=='anomaly').sum()} anomaly, "
               f"{n_masks_found} mask พร้อมใช้)")

    return df_train, df_test, mask_lookup


def load_mask(mask_path: Optional[str], image_size: Tuple[int, int]) -> np.ndarray:
    """โหลด+resize ground-truth mask เป็น binary array {0,1} ขนาด
    image_size — mask_path=None คืน all-zero array (ภาพ normal หรือหา mask
    ไม่เจอ) ใช้ NEAREST resize เสมอ (ไม่ใช่ bilinear) กัน mask เบลอกลายเป็น
    ค่า fraction แปลกๆ ระหว่าง 0-1 ที่ threshold ผิดพลาดได้
    """
    h, w = image_size
    if mask_path is None:
        return np.zeros((h, w), dtype=np.uint8)
    with Image.open(mask_path) as img:
        img = img.convert("L").resize((w, h), Image.NEAREST)
        arr = np.array(img)
    return (arr > 127).astype(np.uint8)


def build_mvtec_loaders(cfg, category: str):
    """สร้าง (train_loader, test_loader, mask_lookup) สำหรับ 1 category —
    reuse AnomalyDataset/build_transforms/make_loader จาก src/data/dataset.py
    ตรงๆ (ดู docstring หัวไฟล์)
    """
    df_train, df_test, mask_lookup = scan_mvtec_category(cfg.MVTEC_ROOT, category)

    imagenet_tf, train_aug_tf, display_tf, preproc_display_tf = build_transforms(cfg)
    norm_tf = train_aug_tf if cfg.USE_AUGMENTATION else imagenet_tf

    train_ds = AnomalyDataset(df_train, norm_tf, display_tf, cfg.IMAGE_SIZE, preproc_display_tf)
    test_ds = AnomalyDataset(df_test, imagenet_tf, display_tf, cfg.IMAGE_SIZE, preproc_display_tf)

    train_loader = make_loader(train_ds, cfg, shuffle=True)
    test_loader = make_loader(test_ds, cfg, shuffle=False)

    return train_loader, test_loader, mask_lookup
