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
"""
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

    # ชี้ไปที่ split_assignment.csv เดียวกับ repo หลักและ PatchCore repo
    # เพื่อให้ train/val/test membership ตรงกันเป๊ะระหว่างทุก baseline —
    # "ตั้งใจไม่ฝัง 'SEED {n}' ไว้ใน path นี้" (ต่างจาก SAVE_PATH/OUTPUT_PATH
    # ด้านล่าง) เพราะอยากให้ทุก seed ใน multi-seed run ใช้ split เดียวกัน
    # เสมอ (วัด variance จาก training/coreset randomness เท่านั้น ไม่ใช่
    # จาก train/val/test membership ที่ต่างกัน) ถ้าต้องการ split แยกต่อ
    # seed ให้ฝัง "SEED 42" ไว้ใน path นี้ด้วย แล้วเพิ่ม
    # 'SPLIT_CACHE_PATH' กลับเข้า TEMPLATE_KEYS ใน RUN_MULTI_SEED.py
    #
    # ⚠️ หมายเหตุ: RUN.py ของ PCB-Anomaly-Baselines-PatchCore ต้นฉบับไม่ได้
    # ฝัง "SEED 42" ไว้ใน SAVE_PATH/OUTPUT_PATH เลย ทำให้ RUN_MULTI_SEED.py
    # ของ repo นั้น raise ValueError ทันทีถ้ารันจริง (marker ที่มันเช็คหา
    # ไม่มีอยู่ใน OVERRIDES) — ไฟล์นี้แก้ไขจุดนั้นแล้วโดยฝัง "SEED 42" ไว้
    # ใน SAVE_PATH/OUTPUT_PATH ให้ครบ แนะนำให้ backport การแก้นี้กลับไปที่
    # PCB-Anomaly-Baselines-PatchCore ด้วย (ดู README)
    SPLIT_CACHE_PATH="splits/split_assignment.csv",
    SAVE_PATH="save/logs/SEED 42",
    OUTPUT_PATH="save/results/SEED 42",
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
