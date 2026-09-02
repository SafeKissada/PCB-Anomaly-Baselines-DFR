"""รัน cost-aware threshold sweep สำหรับ DFR โดยโหลด scores_{split}.npz
ที่ run_dfr.py เซฟไว้แล้ว ไม่รัน inference ใหม่

cost_aware.py รับแค่ scores (float) และ y_true (int 0/1) — ไม่สนว่า score
มาจากวิธีไหน ดังนั้นจึงใช้กับ DFR ได้ทันทีโดยไม่ต้องแก้อะไร (เดียวกับที่
ใช้กับ PatchCore ใน PCB-Anomaly-Baselines-PatchCore)

Uses the same cost_aware.py from the main repo (Anomaly-Detection-THESIS)
directly — it only takes scores (float) and y_true (int 0/1), making it
method-agnostic and immediately usable with DFR's reconstruction-error
scores without any modification.

Usage:
    python scripts/run_cost_aware_dfr.py
    (configure paths in RUN.py, same as run_dfr.py)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config.config import Config


def main():
    """โหลด scores_val.npz + scores_test.npz แล้วรัน cost-aware threshold
    sweep บน val → ประเมินผลบน test → เซฟ cost_aware_sweep.csv ลง SAVE_PATH
    """
    from RUN import OVERRIDES
    from src.cost_aware import cost_sweep_report

    cfg = Config(**OVERRIDES)

    def load(split: str) -> tuple:
        path = Path(cfg.SAVE_PATH) / f"scores_{split}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"ไม่พบ {path} — รัน scripts/run_dfr.py ก่อน\n"
                f"/ {path} not found — run scripts/run_dfr.py first."
            )
        d = np.load(path, allow_pickle=True)
        return d["scores"].astype("float64"), d["y_true"].astype("int64")

    val_scores,  val_y  = load("val")
    test_scores, test_y = load("test")

    r_values = [1, 5, 10, 20, 50, 100]
    report = cost_sweep_report(
        val_scores, val_y, test_scores, test_y, r_values)

    df = pd.DataFrame(report)
    print(df.to_string(index=False))

    out_path = Path(cfg.SAVE_PATH) / "cost_aware_sweep.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
