"""
Smoke test — สร้างภาพ dummy จำนวนน้อยแล้วรัน pipeline เต็ม (fit CAE + score
+ compute_metrics) เพื่อเช็คว่าโค้ดทั้งชุดรันจบไม่มี error ก่อนเอาไปใช้กับ
dataset จริง ซึ่งใช้เวลานานกว่ามาก

ไม่ได้เช็ค correctness ของค่า metric (เพราะภาพ dummy เป็น random noise ไม่มี
สัญญาณ defect จริง) เช็คแค่ว่า "รันได้ไม่ error และ shape ถูกต้อง"

Usage: python tests/smoke_test.py
"""
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import Config
from scripts.run_dfr import run


def make_dummy_dataset(root: Path, n_good=20, n_defect=8, size=(64, 64)):
    good_dir = root / "good"
    defect_dir = root / "defect"
    good_dir.mkdir(parents=True, exist_ok=True)
    defect_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(0)
    for i in range(n_good):
        arr = rng.randint(100, 140, (*size, 3), dtype=np.uint8)  # normal: เทาสม่ำเสมอ
        Image.fromarray(arr).save(good_dir / f"good_{i:03d}.png")
    for i in range(n_defect):
        arr = rng.randint(0, 255, (*size, 3), dtype=np.uint8)  # defect: noise จัด
        Image.fromarray(arr).save(defect_dir / f"defect_{i:03d}.png")


def main():
    tmp_root = Path("/tmp/dfr_smoke_test")
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    data_root = tmp_root / "data"
    make_dummy_dataset(data_root)

    cfg = Config(
        DATA_ROOT=str(data_root),
        SPLIT_CACHE_PATH=str(tmp_root / "splits" / "split_assignment.csv"),
        SAVE_PATH=str(tmp_root / "save/logs"),
        OUTPUT_PATH=str(tmp_root / "save/results"),
        IMAGE_SIZE=(64, 64),
        BATCH_SIZE=4,
        NUM_WORKERS=0,
        BACKBONE="resnet18",           # เบาสุด สำหรับ smoke test เร็วๆ
        PRETRAINED=False,              # sandbox นี้ดาวน์โหลด pretrained weight ไม่ได้ —
                                        # รันจริงต้องเปลี่ยนกลับเป็น True เท่านั้น
        FEATURE_LAYERS=("layer1", "layer2"),
        DFR_ALIGN_MODE="nearest",      # paper default — ดู config/config.py
        DFR_AGG_KERNEL=4,
        DFR_AGG_STRIDE=4,
        DFR_EPOCHS=2,                  # แค่เช็คว่า training loop รันจบไม่ error
        DFR_LATENT_DIM=8,              # fix ค่าตรงๆ กัน PCA ต้องการ sample เยอะเกินไป
        NORMALIZE_FEATURES=False,      # paper default (ไม่ normalize)
        SCORE_METHOD="topk",
        EXPERIMENT="smoke_test",
    )

    val_result, test_result = run(cfg)

    assert val_result.image_scores.shape[0] == len(val_result.labels)
    assert test_result.image_scores.shape[0] == len(test_result.labels)
    assert val_result.pixel_maps.shape[1:] == cfg.IMAGE_SIZE
    assert not np.isnan(val_result.image_scores).any()
    assert not np.isnan(test_result.image_scores).any()

    history_path = Path(cfg.SAVE_PATH) / "history.json"
    assert history_path.exists(), "history.json ต้องถูกสร้าง (DFR มี training loop)"

    print("\n✅ SMOKE TEST PASSED — pipeline รันจบไม่มี error, shape ถูกต้องทั้งหมด")
    shutil.rmtree(tmp_root)


if __name__ == "__main__":
    main()
