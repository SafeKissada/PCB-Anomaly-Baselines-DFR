"""
รัน DFR ซ้ำหลาย seed (multi-seed) — reuse OVERRIDES จาก RUN.py ทุกประการ
ไม่ copy ซ้ำ กัน 2 ไฟล์ไม่ sync กัน

แต่ละ seed รันครบ 3 step เหมือน RUN.py:
  [1/3] run_dfr        — fit + score + save
  [2/3] visualize      — สร้างภาพ
  [3/3] cost_aware     — threshold sweep (toggle จาก RUN.RUN_COST_AWARE_ANALYSIS)

auto-detect "SEED {n}" ใน path ของ TEMPLATE_KEYS แล้วแทนที่ต่อ seed
fail-fast ถ้าไม่เจอ marker ก่อนเริ่ม seed แรก

**GPU memory cleanup ระหว่าง seed (สำคัญ)**: seed แต่ละตัวสร้าง backbone
(frozen, ~100MB-500MB ขึ้นกับ BACKBONE) + CAE ใหม่บน GPU ทุกครั้ง ถ้าไม่
ปล่อย memory ของ seed ก่อนหน้าก่อนเริ่ม seed ถัดไป PyTorch CUDA caching
allocator จะสะสม "reserved but unallocated" memory ไปเรื่อยๆ จน OOM
ทั้งที่แต่ละ seed เดี่ยวๆ ใช้ memory ไม่เยอะขนาดนั้น (อาการนี้สังเกตได้จาก
error message จริงที่เจอ: "20.31 GiB memory in use... 8.51 GiB is
reserved by PyTorch but unallocated") — script นี้จึงเรียก
`del cfg; gc.collect(); torch.cuda.empty_cache()` หลังจบทุก seed
(ทั้งกรณีสำเร็จและ error) เพิ่มเติมจาก cleanup ที่มีอยู่แล้วใน
DFR.fit()/DFR.score() เอง (src/models/dfr.py)

ถ้ายังเจอ OOM หลังจากนี้ ให้ลด `DFR_FEATURE_CHUNK_SIZE` ใน RUN.py
(ค่า default 8 → ลองลดเหลือ 4 หรือ 2) และ/หรือตั้ง environment variable
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ก่อนรัน script นี้ (ทำให้
อัตโนมัติแล้วด้านล่าง ถ้ายังไม่ได้ตั้งไว้เอง)

Each seed runs all 3 steps identical to RUN.py. Auto-detects "SEED {n}"
in TEMPLATE_KEYS paths and substitutes per seed. Fails fast if the marker
is missing before the first seed starts.

**GPU memory cleanup between seeds (important)**: each seed builds a fresh
backbone + CAE on GPU. Without releasing the previous seed's memory before
the next seed starts, PyTorch's CUDA caching allocator accumulates
"reserved but unallocated" memory until OOM — even though any single seed
alone doesn't need that much (this matches the observed error: "20.31 GiB
memory in use... 8.51 GiB is reserved by PyTorch but unallocated"). This
script calls `del cfg; gc.collect(); torch.cuda.empty_cache()` after every
seed (success or failure), on top of the cleanup already inside
`DFR.fit()`/`DFR.score()` (`src/models/dfr.py`).

If OOM still occurs, lower `DFR_FEATURE_CHUNK_SIZE` in `RUN.py` (default 8
→ try 4 or 2) and/or set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
before running this script (done automatically below if not already set).

**ต่างจาก RUN_MULTI_SEED.py ของ PCB-Anomaly-Baselines-PatchCore**:
repo นี้แยก split ต่อ seed ด้วย (TEMPLATE_KEYS รวม SPLIT_CACHE_PATH) ตาม
โครงสร้างที่ต้องการ: seed 1 --> log/table/split, seed 2 --> log/table/split
แยกกันหมดทุก seed — **ต่างจาก default เดิมของ repo นี้เอง** (เวอร์ชันก่อน
หน้า share split เดียวข้าม seed เพื่อ apples-to-apples กับ repo อื่น) ผล
คือ variance ที่วัดได้ข้าม seed ตอนนี้ปนกัน 2 แหล่ง (training randomness +
split-membership ต่างกัน) — ดูคอมเมนต์ใน RUN.py สำหรับ trade-off เต็ม
ถ้าต้องการกลับไป share split เดียวเหมือนเดิม ตัด 'SPLIT_CACHE_PATH' ออกจาก
TEMPLATE_KEYS ด้านล่าง แล้วลบ "SEED 42" ออกจาก SPLIT_CACHE_PATH ใน RUN.py

**Difference from PCB-Anomaly-Baselines-PatchCore's RUN_MULTI_SEED.py**:
this repo templates the split per seed too (`TEMPLATE_KEYS` includes
`SPLIT_CACHE_PATH`), matching the intended layout: seed 1 -->
log/table/split, seed 2 --> log/table/split, all separated per seed —
**different from this repo's own earlier default** (which shared one split
across seeds for apples-to-apples comparison with the other repos). This
means cross-seed variance now mixes two sources (training randomness +
differing split membership) — see the trade-off note in `RUN.py`. To
revert to a single shared split, remove `'SPLIT_CACHE_PATH'` from
`TEMPLATE_KEYS` below and drop `"SEED 42"` from `SPLIT_CACHE_PATH` in
`RUN.py`.
"""
import os
# ตั้งก่อน import torch เพื่อให้มีผลเต็มที่ (ไม่ทับค่าที่ผู้ใช้ตั้งไว้เองแล้ว)
# — ช่วยลด GPU memory fragmentation ระหว่างรันหลาย seed ต่อกันในโปรเซส
# เดียว (ดู PyTorch docs ที่ลิงก์ไว้ใน error message ของ CUDA OOM)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc
import sys
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import RUN
from config.config import Config
import scripts.run_dfr              as run_dfr
import scripts.visualize_dfr        as visualize_dfr
import scripts.run_cost_aware_dfr   as run_cost_aware_dfr

# ── seed ที่จะรัน ────────────────────────────────────────────────────
SEEDS = [1, 14, 42, 63, 123, 228, 450, 1357, 2512, 19999]

# ── key ที่ต้องแยกตาม seed ──────────────────────────────────────────
# รวม SPLIT_CACHE_PATH ด้วย ตามโครงสร้างที่ต้องการ (seed แต่ละตัวมี
# log/table/split เป็นของตัวเอง) — ดูคอมเมนต์หัวไฟล์สำหรับ trade-off
TEMPLATE_KEYS = ['SAVE_PATH', 'OUTPUT_PATH', 'SPLIT_CACHE_PATH']

# ── auto-detect template จาก OVERRIDES ─────────────────────────────
_current_seed = RUN.OVERRIDES['SEED']
_marker = f'SEED {_current_seed}'
_placeholder = 'SEED {seed}'

path_templates = {}
for key in TEMPLATE_KEYS:
    original_value = RUN.OVERRIDES[key]
    if _marker not in original_value:
        raise ValueError(
            f"ไม่เจอ '{_marker}' ใน RUN.OVERRIDES['{key}'] "
            f"(= {original_value!r}) — ต้องฝัง '{_marker}' ไว้ใน path "
            f"ให้ script แทนที่ด้วย seed อื่นได้\n"
            f"/ '{_marker}' not found in RUN.OVERRIDES['{key}'] "
            f"(= {original_value!r}) — embed '{_marker}' in the path "
            f"so this script can substitute other seeds.")
    path_templates[key] = original_value.replace(_marker, _placeholder)

print("Path templates ที่ตรวจพบ:")
for key, tmpl in path_templates.items():
    print(f"  {key} = {tmpl!r}")

results_log = []

for i, seed in enumerate(SEEDS, start=1):
    _n_steps = 3 if RUN.RUN_COST_AWARE_ANALYSIS else 2

    print(f"\n{'=' * 70}")
    print(f" MULTI-SEED RUN [{i}/{len(SEEDS)}] — SEED={seed}")
    print(f"{'=' * 70}")

    RUN.OVERRIDES['SEED'] = seed
    for key, tmpl in path_templates.items():
        RUN.OVERRIDES[key] = tmpl.format(seed=seed)

    print(f"  SAVE_PATH        -> {RUN.OVERRIDES['SAVE_PATH']}")
    print(f"  OUTPUT_PATH      -> {RUN.OVERRIDES['OUTPUT_PATH']}")
    print(f"  SPLIT_CACHE_PATH -> {RUN.OVERRIDES['SPLIT_CACHE_PATH']}")

    try:
        cfg = Config(**RUN.OVERRIDES)

        print(f"\n  --- [1/{_n_steps}] fit + score + save ---")
        run_dfr.run(cfg)

        print(f"\n  --- [2/{_n_steps}] visualize ---")
        visualize_dfr.visualize(cfg)

        if RUN.RUN_COST_AWARE_ANALYSIS:
            print(f"\n  --- [3/{_n_steps}] cost-aware sweep ---")
            run_cost_aware_dfr.main()

        results_log.append((seed, 'OK', None))
        print(f"\n  ✅ seed={seed} เสร็จ -> {RUN.OVERRIDES['SAVE_PATH']}")

    except Exception as e:
        results_log.append((seed, 'FAILED', str(e)))
        print(f"\n  ❌ seed={seed} ล้มเหลว: {e}")
        traceback.print_exc()
        print("  ข้าม seed นี้ ไปทำ seed ถัดไปต่อ...")
        continue

    finally:
        # ปล่อย GPU memory ของ seed นี้ก่อนเริ่ม seed ถัดไปเสมอ (ทั้งกรณี
        # สำเร็จและ error) — ดู docstring หัวไฟล์ว่าทำไมจำเป็น
        try:
            del cfg
        except NameError:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ── สรุปผล ─────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print(" สรุปผล Multi-Seed Run")
print(f"{'=' * 70}")
for seed, status, err in results_log:
    line = f"  seed={seed:<4}  {status}"
    if err:
        line += f"  ({err})"
    print(line)

n_ok = sum(1 for _, s, _ in results_log if s == 'OK')
print(f"\nสำเร็จ {n_ok}/{len(SEEDS)} seed")
if n_ok < len(SEEDS):
    print("⚠️  เช็ค traceback ด้านบนก่อนเอาผลไปสรุปสถิติ")
