"""
รัน DFR ซ้ำหลาย seed (multi-seed) — reuse OVERRIDES จาก RUN.py ทุกประการ
ไม่ copy ซ้ำ กัน 2 ไฟล์ไม่ sync กัน

แต่ละ seed รันครบ 3 step เหมือน RUN.py:
  [1/3] run_dfr        — fit + score + save
  [2/3] visualize      — สร้างภาพ
  [3/3] cost_aware     — threshold sweep (toggle จาก RUN.RUN_COST_AWARE_ANALYSIS)

auto-detect "SEED {n}" ใน path ของ TEMPLATE_KEYS แล้วแทนที่ต่อ seed
fail-fast ถ้าไม่เจอ marker ก่อนเริ่ม seed แรก

**ต่างจาก RUN_MULTI_SEED.py ของ PCB-Anomaly-Baselines-PatchCore**:
TEMPLATE_KEYS ที่นี่ = ['SAVE_PATH', 'OUTPUT_PATH'] เท่านั้น (ไม่รวม
SPLIT_CACHE_PATH) เพราะ RUN.py ของ repo นี้ตั้งใจไม่ฝัง "SEED 42" ไว้ใน
SPLIT_CACHE_PATH (ดูคอมเมนต์ใน RUN.py) — ทุก seed จึงใช้ train/val/test
split เดียวกันเสมอ วัด variance จาก training randomness (CAE weight init
+ batch order) เท่านั้น ไม่ปนกับ variance จาก split ที่ต่างกัน ถ้าต้องการ
พฤติกรรมแบบ PatchCore repo (แยก split ต่อ seed ด้วย) ให้เพิ่ม "SEED 42"
เข้า SPLIT_CACHE_PATH ใน RUN.py แล้วเพิ่ม 'SPLIT_CACHE_PATH' กลับเข้า
TEMPLATE_KEYS ด้านล่าง

Each seed runs all 3 steps identical to RUN.py. Auto-detects "SEED {n}"
in TEMPLATE_KEYS paths and substitutes per seed. Fails fast if the marker
is missing before the first seed starts.

**Difference from PCB-Anomaly-Baselines-PatchCore's RUN_MULTI_SEED.py**:
TEMPLATE_KEYS here is ['SAVE_PATH', 'OUTPUT_PATH'] only (no
SPLIT_CACHE_PATH), because this repo's RUN.py intentionally does not embed
"SEED 42" into SPLIT_CACHE_PATH (see the comment in RUN.py) — every seed
therefore shares the same train/val/test split, so measured variance
reflects training randomness (CAE weight init + batch order) only, not
mixed in with split-membership variance. To match the PatchCore repo's
behavior (a separate split per seed too), add "SEED 42" to
SPLIT_CACHE_PATH in RUN.py and add 'SPLIT_CACHE_PATH' back to
TEMPLATE_KEYS below.
"""
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import RUN
from config.config import Config
import scripts.run_dfr              as run_dfr
import scripts.visualize_dfr        as visualize_dfr
import scripts.run_cost_aware_dfr   as run_cost_aware_dfr

# ── seed ที่จะรัน ────────────────────────────────────────────────────
SEEDS = [1, 14, 42, 63, 123, 228, 450, 1357, 2512, 19999]

# ── key ที่ต้องแยกตาม seed (ดูหมายเหตุด้านบนว่าทำไมไม่รวม SPLIT_CACHE_PATH) ──
TEMPLATE_KEYS = ['SAVE_PATH', 'OUTPUT_PATH']

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
print(f"  SPLIT_CACHE_PATH (shared, ไม่แยกต่อ seed) = {RUN.OVERRIDES['SPLIT_CACHE_PATH']!r}")

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
