"""
รัน DFR แบบ end-to-end บน dataset เดียวกับ repo หลัก
(Anomaly-Detection-THESIS) — metric ที่ได้เทียบกับ EXPERIMENT 0
(ConvNeXt+AE) และ PatchCore ได้โดยตรงเพราะใช้ split/evaluate เดียวกัน

output แบ่งเป็น 2 โฟลเดอร์:
  SAVE_PATH   — ตัวเลข/log: scores_{split}.npz, final_results_{split}.json,
                roc_curve_data_{split}.csv, history.json, README.md
  OUTPUT_PATH — ภาพ: (สร้างโดย scripts/visualize_dfr.py)

ฟังก์ชัน run() ถูกเรียกได้จาก:
  RUN.py            → รันรอบเดียว
  RUN_MULTI_SEED.py → รันซ้ำหลาย seed โดย reuse OVERRIDES เดิม

Runs DFR end-to-end on the same dataset as the main repo
(Anomaly-Detection-THESIS) — metrics compare directly against EXPERIMENT 0
(ConvNeXt+AE) and PatchCore since they share the same split/evaluate code.
"""
import gc
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import Config, set_seed
from src.data.dataset import build_datasets_and_loaders
from src.evaluate import (compute_metrics, select_percentile_threshold,
                           compute_naive_baseline_metrics)
from src.io_utils import (save_final_results, save_scores,
                           save_roc_csv, save_training_history,
                           write_save_path_readme,
                           compute_heatmap_calibration,
                           normalize_pixel_maps_global)
from src.models.dfr import DFR

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("run_dfr")


def run(cfg: Config):
    """รัน DFR 1 รอบเต็ม: split → fit CAE → score → threshold → save

    split caching ทำงานอัตโนมัติผ่าน build_datasets_and_loaders() —
    รอบแรก: compute + save cache ลง cfg.SPLIT_CACHE_PATH
    รอบถัดไป: โหลด cache โดยตรง ไม่ compute ซ้ำ

    Split caching is automatic via build_datasets_and_loaders():
    first run computes and saves the cache; subsequent runs load it directly.
    """
    set_seed(cfg.SEED)

    logger.info(
        f"Loading data from {cfg.DATA_ROOT} "
        f"(split cache: {cfg.SPLIT_CACHE_PATH})"
    )
    data = build_datasets_and_loaders(cfg)
    logger.info(
        f"Train (normal only): {len(data['df_train'])} images | "
        f"Val: {len(data['df_val'])} | Test: {len(data['df_test'])}"
    )

    model = DFR(cfg)
    model.fit(data["normal_loader"])

    # เซฟ training history ก่อน score() เผื่อ score() พังกลางทาง ยังมี
    # ประวัติการเทรนไว้ debug ได้ (ต่างจาก PatchCore/PaDiM ที่ไม่มี
    # ฟังก์ชันนี้เพราะไม่มี training loop)
    save_training_history(cfg, model.history)

    val_result  = model.score(data["val_loader"])
    test_result = model.score(data["test_loader"])

    # y_true (int 0/1) ใช้คำนวณ metric เสมอ — ไม่ใช้ result.labels (string)
    # ซึ่งเป็นชื่อ class สำหรับ display เท่านั้น (ดู src/models/base.py)
    threshold = select_percentile_threshold(
        val_result.image_scores, val_result.y_true, cfg)
    logger.info(
        f"Threshold (percentile={cfg.THRESHOLD_PERCENTILE}): {threshold:.6f}"
    )

    # ── Global heatmap calibration ──────────────────────────────────────
    # pixel_maps ตอนนี้ยังเป็น raw L2 reconstruction-error (ดู
    # DFR.score() ใน src/models/dfr.py) — src/visual.py (overlay_heatmap)
    # สมมติว่า input อยู่ในช่วง [0,1] เสมอ ถ้าไม่ normalize ก่อน แทบทุก
    # pixel จะโดน clip เป็น 1.0 (บั๊กเดียวกับที่เจอใน PatchCore ตอน sync
    # ครั้งก่อน — ดู src/io_utils.py::compute_heatmap_calibration)
    heatmap_vmin, heatmap_vmax = compute_heatmap_calibration(
        val_result.pixel_maps, val_result.y_true)
    logger.info(
        f"Heatmap calibration (global, val-normal percentiles): "
        f"vmin={heatmap_vmin:.6f}  vmax={heatmap_vmax:.6f}"
    )
    val_result.pixel_maps = normalize_pixel_maps_global(
        val_result.pixel_maps, heatmap_vmin, heatmap_vmax)
    test_result.pixel_maps = normalize_pixel_maps_global(
        test_result.pixel_maps, heatmap_vmin, heatmap_vmax)

    # คำนวณ naive baseline บน val และ test แยกกัน — ใช้ cfg.SEED เดียวกัน
    # กับ pipeline ทั้งหมด เพื่อให้ random_prior reproduce ได้ข้าม run
    naive = {
        'val':  compute_naive_baseline_metrics(val_result.y_true,  cfg.SEED),
        'test': compute_naive_baseline_metrics(test_result.y_true, cfg.SEED),
    }

    extra_fields = {
        "dfr_epochs": cfg.DFR_EPOCHS,
        "dfr_lr": cfg.DFR_LR,
        "latent_dim": model.latent_dim,
        "regional_feature_channels": model.c_in,
        "regional_feature_spatial_shape": list(model.embed_spatial_shape),
        "normalize_features": cfg.NORMALIZE_FEATURES,
        "score_method": cfg.SCORE_METHOD,
    }

    for split_name, result in [("val", val_result), ("test", test_result)]:
        metrics = compute_metrics(
            result.image_scores, result.y_true, threshold)
        logger.info(
            f"[{split_name}] AUC={metrics['auc']:.4f}  "
            f"AP={metrics['ap']:.4f}  "
            f"EscapeRate={metrics['escape_rate']:.4f}  "
            f"AutoClearRate={metrics['auto_clear_rate']:.4f}  "
            f"F1={metrics['f1']:.4f}"
        )

        # ── SAVE_PATH: ตัวเลข/log ─────────────────────────────────────
        save_scores(
            cfg, split_name,
            result.image_scores, result.y_true,
            result.labels, result.paths,
            result.pixel_maps, result.orig_imgs, result.preproc_imgs,
        )
        save_final_results(cfg, split_name, metrics, threshold,
                           naive_baselines=naive[split_name],
                           extra_fields=extra_fields)
        save_roc_csv(cfg, split_name,
                     result.image_scores, result.y_true)

    # README เขียนท้ายสุดหลังทุก split เสร็จ เพื่อให้ list ไฟล์ครบ
    write_save_path_readme(cfg)

    # ปล่อย GPU memory ของ backbone+CAE ทันทีที่ใช้เสร็จ — สำคัญเวลาเรียก
    # run() ซ้ำหลายรอบในโปรเซสเดียว (เช่น จาก RUN_MULTI_SEED.py) ไม่งั้น
    # CUDA caching allocator จะสะสม reserved memory ไปเรื่อยๆ จน OOM ทั้งที่
    # แต่ละรอบเดี่ยวๆ ใช้ memory ไม่เยอะขนาดนั้น (ดู RUN_MULTI_SEED.py
    # หัวไฟล์สำหรับรายละเอียดเต็ม)
    del model
    gc.collect()
    if cfg.DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    logger.info(
        f"All artifacts saved → SAVE_PATH: {cfg.SAVE_PATH}"
    )
    return val_result, test_result


if __name__ == "__main__":
    cfg = Config()
    run(cfg)
