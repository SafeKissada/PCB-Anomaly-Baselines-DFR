"""
Smoke test สำหรับ MVTec AD pipeline ทั้งชุด (loader -> DFR.fit/score ->
mask lookup -> pixel_roc_auc/pro_auc) — สร้างโฟลเดอร์ปลอมที่มีโครงสร้าง
เหมือน MVTec AD จริงทุกประการ (train/test/ground_truth) แต่ใช้ภาพสังเคราะห์
(ไม่ใช่ MVTec จริง เพราะ sandbox นี้ดาวน์โหลดไม่ได้ — ดู README)

ยืนยันได้แค่ "pipeline เดินถูกโครงสร้าง ไม่ error" — ไม่ยืนยันว่าตัวเลข
ROC-AUC/PRO-AUC ที่ได้จริงจาก MVTec AD จะตรงกับ paper แค่ไหน (ต้องรันกับ
ข้อมูลจริงเองถึงจะรู้)

Usage: python tests/mvtec_pipeline_test.py
"""
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.mvtec_config import MVTecConfig
from scripts.run_dfr_mvtec import main as run_mvtec_main


def make_fake_mvtec_category(root: Path, category: str, img_size=64,
                              n_train=16, n_test_good=6, n_test_defect=6,
                              seed=0):
    """สร้าง 1 category ปลอมตามโครงสร้าง MVTec จริง:
        {root}/{category}/train/good/*.png
        {root}/{category}/test/good/*.png
        {root}/{category}/test/scratch/*.png
        {root}/{category}/ground_truth/scratch/*_mask.png
    """
    rng = np.random.RandomState(seed)
    cat_root = root / category
    (cat_root / "train" / "good").mkdir(parents=True, exist_ok=True)
    (cat_root / "test" / "good").mkdir(parents=True, exist_ok=True)
    (cat_root / "test" / "scratch").mkdir(parents=True, exist_ok=True)
    (cat_root / "ground_truth" / "scratch").mkdir(parents=True, exist_ok=True)

    def make_bg():
        return 120 + rng.randint(-8, 8, (img_size, img_size, 1))

    for i in range(n_train):
        arr = np.clip(np.repeat(make_bg(), 3, axis=2), 0, 255).astype(np.uint8)
        Image.fromarray(arr).save(cat_root / "train" / "good" / f"{i:03d}.png")

    for i in range(n_test_good):
        arr = np.clip(np.repeat(make_bg(), 3, axis=2), 0, 255).astype(np.uint8)
        Image.fromarray(arr).save(cat_root / "test" / "good" / f"{i:03d}.png")

    for i in range(n_test_defect):
        arr = np.clip(np.repeat(make_bg(), 3, axis=2), 0, 255).astype(np.uint8)
        y0 = int(rng.randint(4, img_size - 20))
        x0 = int(rng.randint(4, img_size - 20))
        arr[y0:y0 + 16, x0:x0 + 16, :] = 250
        Image.fromarray(arr).save(cat_root / "test" / "scratch" / f"{i:03d}.png")

        mask = np.zeros((img_size, img_size), dtype=np.uint8)
        mask[y0:y0 + 16, x0:x0 + 16] = 255
        Image.fromarray(mask).save(
            cat_root / "ground_truth" / "scratch" / f"{i:03d}_mask.png")


def main():
    tmp_root = Path("/tmp/mvtec_pipeline_test")
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    mvtec_root = tmp_root / "mvtec_fake"
    make_fake_mvtec_category(mvtec_root, "fake_cat_1", seed=0)
    make_fake_mvtec_category(mvtec_root, "fake_cat_2", seed=1)

    cfg = MVTecConfig(
        MVTEC_ROOT=str(mvtec_root),
        CATEGORIES=("fake_cat_1", "fake_cat_2"),
        SAVE_PATH=str(tmp_root / "save/logs"),
        OUTPUT_PATH=str(tmp_root / "save/table"),
        IMAGE_SIZE=(64, 64),
        BATCH_SIZE=4,
        NUM_WORKERS=0,
        BACKBONE="resnet18",           # เบา สำหรับ offline smoke test เท่านั้น
        PRETRAINED=False,              # sandbox นี้โหลด pretrained weight ไม่ได้ —
                                        # รันจริงต้องเปลี่ยนกลับเป็น True เท่านั้น
        FEATURE_LAYERS=("layer1", "layer2"),
        DFR_EPOCHS=3,                  # แค่เช็ค pipeline เดินจบ ไม่ใช่ reproduce จริง
        DFR_LATENT_DIM=8,
        DFR_REFLECTION_PADDING=True,
        PRO_NUM_THRESHOLDS=30,         # ลดจาก 200 ให้เร็วขึ้นสำหรับ smoke test
        EXPERIMENT="mvtec_pipeline_smoke_test",
    )

    df = run_mvtec_main(cfg)

    assert len(df) == 2, f"Expected 2 categories, got {len(df)}"
    assert not df["roc_auc"].isna().all(), "ROC-AUC เป็น NaN ทุก category — pipeline พัง"
    assert (df["n_test"] == 12).all(), f"Expected 12 test images/category, got {df['n_test'].tolist()}"
    assert (df["n_test_anomaly"] == 6).all(), (
        f"Expected 6 anomaly test images/category, got {df['n_test_anomaly'].tolist()}")

    print("\n✅ MVTEC PIPELINE SMOKE TEST PASSED — "
         "loader + DFR.fit/score + mask lookup + pixel-ROC-AUC/PRO-AUC "
         "ทั้งสายรันจบไม่มี error, shape/count ถูกต้องทั้งหมด")

    shutil.rmtree(tmp_root)


if __name__ == "__main__":
    main()
