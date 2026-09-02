"""
One-shot entry point สำหรับ DFR — แก้ OVERRIDES แล้วรัน:
    python RUN.py

รันครบ 3 step ในคำสั่งเดียว เหมือน RUN.py ของ repo หลัก และของ
PCB-Anomaly-Baselines-PatchCore:
  [1/3] run_dfr        — fit CAE + score + save .npz/.json/.csv/history.json
  [2/3] visualize      — สร้างภาพทุกใบจาก .npz ที่เพิ่งเซฟ (รวม training_history.png)
  [3/3] cost_aware     — threshold sweep (ปิดได้ผ่าน toggle ด้านล่าง)

OVERRIDES เป็นเพียงที่เดียวที่ต้องแก้ — RUN_MULTI_SEED.py import
OVERRIDES จากไฟล์นี้โดยตรง ไม่ copy ซ้ำ กัน 2 ไฟล์ไม่ sync กัน

ถ้าเจอ CUDA OOM ระหว่างรัน (โดยเฉพาะกับ VGG19 front-12/16 + BATCH_SIZE
ใหญ่): ลด DFR_FEATURE_CHUNK_SIZE ด้านล่างก่อน (default 8 → ลองเหลือ 4
หรือ 2) แทนการลด BATCH_SIZE เอง — ดู docstring ของ
src/models/dfr.py::_regional_feature_map สำหรับสาเหตุเต็ม
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.config import Config
import scripts.run_dfr              as run_dfr
import scripts.visualize_dfr        as visualize_dfr
import scripts.run_cost_aware_dfr   as run_cost_aware_dfr

# ── เปิด/ปิด step [3/3] ─────────────────────────────────────────────
RUN_COST_AWARE_ANALYSIS = True

OVERRIDES = dict(
    # ── Data & paths — แก้ก่อนรันจริง ──────────────────────────────
    DATA_ROOT="dataset root path (contains good/ and defect/ subfolders)",
    GOOD_DIRNAME="good",
    DEFECT_DIRNAME="defect",

    # โครงสร้างที่ต้องการ: seed 1 --> log/table/split, seed 2 --> log/table/split
    # แยกกันหมดทุก seed (log, table, "และ" split cache) — ไม่ share
    # split_assignment.csv ข้าม seed แบบ default เดิมอีกต่อไป
    #
    # **Trade-off ที่ควรรู้ก่อนตัดสินใจแบบนี้** (เขียนไว้ให้ชัด เพราะเป็นจุด
    # ที่กระทบการตีความผลจริง): variance ที่วัดได้ข้าม seed ตอนนี้จะปนกัน
    # 2 แหล่ง — (1) training randomness (CAE weight init/batch order) และ
    # (2) train/val/test membership ที่ต่างกันในแต่ละ seed (เพราะ
    # SPLIT_CACHE_PATH ถูก re-compute ใหม่ทุก seed ด้วย seed นั้นๆ) แยกไม่
    # ออกว่าตัวเลขที่ต่างกันมาจากอะไร และเทียบ per-seed กับ
    # Anomaly-Detection-THESIS/PatchCore repo ตรงๆ ไม่ได้อีกต่อไป (เว้นแต่
    # repo อื่นจะ regenerate split ด้วย seed เดียวกันแบบนี้เหมือนกันทุกที่)
    # ถ้าต้องการเทียบ metric เฉลี่ยรวม (mean±std ข้าม seed) เพื่อดู
    # "ความเสถียรของ method" อย่างเดียว ไม่ได้สนใจเทียบ per-image กับ
    # baseline อื่น แบบนี้ก็ใช้ได้ปกติ
    SPLIT_CACHE_PATH="save/SEED 42/split/split_assignment.csv",
    SAVE_PATH="save/SEED 42/log",
    OUTPUT_PATH="save/SEED 42/table",
    SEED=42,

    # ── Model config — ปรับได้ตามต้องการ ────────────────────────────
    # ค่า default ทั้งหมดด้านล่างนี้ "ตรงตาม paper" (VGG19 front-12,
    # nearest-neighbor align, mean-filter aggregate 4/4, ไม่ normalize) —
    # ไม่ได้ override ไปจาก config/config.py เลย เขียนไว้ตรงนี้ซ้ำเพื่อให้
    # เห็นชัดว่าอยากรันแบบไหน ไม่ต้องเปิด config.py ไปดู ถ้าจะสลับไปเทียบ
    # backbone เดียวกับ EXPERIMENT 0/PatchCore/PaDiM แทน ดูตัวอย่าง
    # ConvNeXt variant ที่ comment ไว้ด้านล่าง
    EXPERIMENT="DFR_group1_VGG19_paper_faithful",
    BACKBONE="vgg19",
    FEATURE_LAYERS=(
        "features.1",  "features.3",  "features.6",  "features.8",
        "features.11", "features.13", "features.15", "features.17",
        "features.20", "features.22", "features.24", "features.26",
    ),  # front-12 ของ VGG19 (paper section IV-C.1) — ดู
        # src.models.dfr.VGG19_ALL_16_LAYERS ถ้าต้องการครบ 16 scale เต็ม
    DFR_ALIGN_MODE="nearest",
    DFR_AGG_KERNEL=4,
    DFR_AGG_STRIDE=4,
    NORMALIZE_FEATURES=False,
    DFR_EPOCHS=100,
    DFR_LR=1e-4,
    THRESHOLD_PERCENTILE=95.0,
    # จำกัด peak GPU memory ของขั้น align (paper eq. 1) — ดู docstring ของ
    # src/models/dfr.py::_regional_feature_map ถ้าเจอ CUDA OOM ให้ลดค่านี้
    # ลงก่อน (เช่น 4 หรือ 2) แทนการลด BATCH_SIZE เอง
    DFR_FEATURE_CHUNK_SIZE=8,

    # ── ConvNeXt variant (apples-to-apples กับ EXPERIMENT 0/PatchCore/PaDiM) ──
    # ยังไม่ฟันธงว่า thesis จะใช้แบบไหน — ถ้าตัดสินใจแล้วอยากสลับ ลบ 6 key
    # ด้านบน (BACKBONE ถึง NORMALIZE_FEATURES) ออก แล้วแทนด้วย:
    #   BACKBONE="convnext_tiny",
    #   FEATURE_LAYERS=("features.3", "features.5"),
    #   DFR_ALIGN_MODE="bilinear",
    #   NORMALIZE_FEATURES=True,
    # (DFR_AGG_KERNEL/DFR_AGG_STRIDE ยังใช้ค่า default 4/4 ได้ตามปกติ)
)

if __name__ == "__main__":
    _n_steps = 3 if RUN_COST_AWARE_ANALYSIS else 2
    cfg = Config(**OVERRIDES)

    print(f"\n--- [1/{_n_steps}] DFR: fit + score + save ---")
    run_dfr.run(cfg)

    print(f"\n--- [2/{_n_steps}] Visualize: สร้างภาพทั้งหมด ---")
    visualize_dfr.visualize(cfg)

    if RUN_COST_AWARE_ANALYSIS:
        print(f"\n--- [3/{_n_steps}] Cost-Aware Threshold Sweep ---")
        run_cost_aware_dfr.main()

    print("\n✅ เสร็จสิ้นกระบวนการทั้งหมดเรียบร้อย!")
