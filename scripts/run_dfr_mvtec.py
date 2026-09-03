"""
รัน DFR บน MVTec AD จริง (reproduce paper Table II/III) — ต่างจาก
scripts/run_dfr.py (PCB thesis) ตรงที่:
  - ใช้ train/test split ที่มากับ MVTec เอง ไม่สุ่มแบ่งเอง
  - loop ทุก category (หรือบางส่วนตาม cfg.CATEGORIES)
  - metric เป็น pixel-ROC-AUC + PRO-AUC (ระดับ pixel/region) ไม่ใช่
    classification metric ระดับภาพแบบ scripts/run_dfr.py

**ก่อนรันสคริปต์นี้ต้องดาวน์โหลด+แตกไฟล์ MVTec AD เองก่อน** — sandbox ที่ใช้
พัฒนา repo นี้ไม่มีสิทธิ์เข้าถึงโดเมนที่ใช้ดาวน์โหลด MVTec AD ได้ (ดู README
หัวข้อ "MVTec AD reproduction" สำหรับลิงก์และเหตุผลเต็ม) แก้ MVTEC_ROOT ใน
OVERRIDES ด้านล่างให้ชี้ไปที่โฟลเดอร์ที่แตกแล้ว

Usage:
    python scripts/run_dfr_mvtec.py
"""
import logging
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config.config import set_seed  # pure utility (random/np/torch seeding), ไม่ผูกกับ PCB fields ใดๆ
from config.mvtec_config import MVTecConfig, MVTEC_ALL_CATEGORIES
from src.data.mvtec_dataset import build_mvtec_loaders, discover_categories, load_mask
from src.mvtec_evaluate import pixel_roc_auc, pro_auc
from src.models.dfr import DFR

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("run_dfr_mvtec")


def run_category(cfg: MVTecConfig, category: str) -> dict:
    """เทรน+ประเมิน DFR บน 1 category คืน dict ที่มี roc_auc/pro_auc/เวลาที่ใช้"""
    t0 = time.time()
    train_loader, test_loader, mask_lookup = build_mvtec_loaders(cfg, category)

    model = DFR(cfg)
    model.fit(train_loader)
    result = model.score(test_loader)

    # เรียง mask ให้ตรงกับลำดับ result.paths เป๊ะ (DataLoader ไม่ shuffle
    # test set อยู่แล้ว แต่ยึดตาม paths ตรงๆ กันพลาดเรื่อง ordering)
    masks = [load_mask(mask_lookup.get(p), cfg.IMAGE_SIZE) for p in result.paths]
    import numpy as np
    masks = np.stack(masks, axis=0)  # [N, H, W] uint8 {0,1}

    roc = pixel_roc_auc(masks, result.pixel_maps)
    pro = pro_auc(masks, result.pixel_maps, max_fpr=cfg.PRO_MAX_FPR,
                  num_thresholds=cfg.PRO_NUM_THRESHOLDS)

    elapsed = time.time() - t0
    logger.info(f"[{category}] ROC-AUC={roc:.4f}  PRO-AUC={pro:.4f}  "
               f"(n_test={len(result.paths)}, latent_dim={model.latent_dim}, "
               f"{elapsed:.1f}s)")

    return dict(
        category=category,
        roc_auc=roc,
        pro_auc=pro,
        n_train=len(train_loader.dataset),
        n_test=len(test_loader.dataset),
        n_test_anomaly=int((masks.reshape(len(masks), -1).sum(axis=1) > 0).sum()),
        latent_dim=model.latent_dim,
        c_in=model.c_in,
        elapsed_sec=round(elapsed, 1),
    )


def main(cfg: MVTecConfig = None):
    if cfg is None:
        cfg = MVTecConfig()

    set_seed(cfg.SEED)

    categories = cfg.CATEGORIES or discover_categories(cfg.MVTEC_ROOT, MVTEC_ALL_CATEGORIES)
    logger.info(f"จะรัน {len(categories)} category: {list(categories)}")
    logger.info(f"BACKBONE={cfg.BACKBONE}  FEATURE_LAYERS={cfg.FEATURE_LAYERS}  "
               f"IMAGE_SIZE={cfg.IMAGE_SIZE}  DFR_EPOCHS={cfg.DFR_EPOCHS}")

    rows = []
    for i, category in enumerate(categories, start=1):
        logger.info(f"\n{'=' * 70}\n[{i}/{len(categories)}] Category: {category}\n{'=' * 70}")
        try:
            rows.append(run_category(cfg, category))
        except Exception as e:
            logger.error(f"[{category}] ล้มเหลว: {e}")
            traceback.print_exc()
            rows.append(dict(category=category, roc_auc=float("nan"),
                             pro_auc=float("nan"), error=str(e)))
            logger.warning(f"ข้าม {category} ไปทำ category ถัดไปต่อ...")
            continue

    df = pd.DataFrame(rows)

    # ── สรุปแบบเดียวกับ paper Table II/III (แยก textures/objects + mean) ──
    textures = {"carpet", "grid", "leather", "tile", "wood"}
    df["group"] = df["category"].apply(lambda c: "texture" if c in textures else "object")

    print("\n" + "=" * 70)
    print(" ผลลัพธ์แยกตาม category")
    print("=" * 70)
    print(df[["category", "group", "roc_auc", "pro_auc", "n_test",
             "n_test_anomaly", "elapsed_sec"]].to_string(index=False))

    print("\n" + "=" * 70)
    print(" สรุปเฉลี่ย (เทียบรูปแบบกับ paper TABLE II/III)")
    print("=" * 70)
    for group_name, group_df in df.groupby("group"):
        print(f"  {group_name:10s}  ROC-AUC={group_df['roc_auc'].mean():.4f}  "
             f"PRO-AUC={group_df['pro_auc'].mean():.4f}  (n={len(group_df)})")
    print(f"  {'overall':10s}  ROC-AUC={df['roc_auc'].mean():.4f}  "
         f"PRO-AUC={df['pro_auc'].mean():.4f}  (n={len(df)})")

    out_path = Path(cfg.SAVE_PATH) / "mvtec_results.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"\nบันทึกผลลัพธ์เต็ม -> {out_path}")

    n_failed = df["roc_auc"].isna().sum()
    if n_failed > 0:
        logger.warning(f"⚠️  {n_failed}/{len(df)} category ล้มเหลว — เช็ค log "
                       f"ด้านบนก่อนเอาค่าเฉลี่ยไปอ้างอิง (ค่าเฉลี่ยข้างบนคำนวณ "
                       f"จาก category ที่เหลือเท่านั้น pandas .mean() ข้าม NaN "
                       f"อัตโนมัติ)")

    return df


if __name__ == "__main__":
    main()
