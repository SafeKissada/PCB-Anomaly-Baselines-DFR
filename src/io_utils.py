"""I/O utilities สำหรับ DFR baseline — ออกแบบให้ schema ของ output ตรงกับ
repo หลัก (Anomaly-Detection-THESIS) และ PCB-Anomaly-Baselines-PatchCore
ทุก key เพื่อให้ script เทียบผลข้าม repo ใช้ชื่อไฟล์และ key เดียวกันได้โดย
ไม่มี silent mismatch

**หมายเหตุความต่างจาก io_utils.py ของ PCB-Anomaly-Baselines-PatchCore**:
ต้นฉบับ hardcode `"method": "PatchCore"` และ `"coreset_ratio": cfg.CORESET_RATIO`
ไว้ตรงๆ ใน save_final_results() ซึ่งเป็น field เฉพาะของ PatchCore ทำให้เอาไป
reuse ตรงๆ กับ method อื่นไม่ได้ (จะ error เพราะ DFR config ไม่มี
CORESET_RATIO, และ JSON จะโกหกว่า method="PatchCore" ทั้งที่รัน DFR)
ที่นี่แก้ 2 จุดนั้นให้ generic:
  1. "method": cfg.METHOD_NAME  (Config ทุกตัวต้องมี field นี้)
  2. "coreset_ratio" คงที่ → extra_fields: dict ที่ผู้เรียกส่งเข้ามาเอง
     (ไม่ส่งก็ได้ ไม่ error, ค่า default คือไม่มี key นี้เลย)
ทั้งสองจุดเป็น backward-compatible change (parameter ใหม่มี default)
ไม่กระทบ call site เดิมถ้ามีการ reuse ไฟล์นี้แบบเดิม — แนะนำให้ backport
กลับไปที่ PCB-Anomaly-Baselines-PatchCore ด้วยเพื่อความสอดคล้องกันทั้ง
ระบบ (ดู README หัวข้อ "ความต่างจาก PCB-Anomaly-Baselines-PatchCore")

การแบ่ง SAVE_PATH / OUTPUT_PATH:
  SAVE_PATH   — ตัวเลข/log: .npz, .json, .csv  (โหลดกลับคำนวณต่อได้)
  OUTPUT_PATH — ภาพ: .png เท่านั้น             (ดูด้วยตาเท่านั้น)

I/O utilities for the DFR baseline — designed so output schemas match the
main repo (Anomaly-Detection-THESIS) and PCB-Anomaly-Baselines-PatchCore
key-for-key, letting cross-repo comparison scripts use the same filenames
and key names without any silent mismatch.

**Note on the difference from PCB-Anomaly-Baselines-PatchCore's io_utils.py**:
the original hardcodes `"method": "PatchCore"` and
`"coreset_ratio": cfg.CORESET_RATIO` directly inside save_final_results(),
which are PatchCore-only fields — reusing that file as-is for another
method would break (DFR's config has no CORESET_RATIO, and the JSON would
falsely claim method="PatchCore" while actually running DFR). This version
fixes both:
  1. "method": cfg.METHOD_NAME  (every Config must define this field)
  2. the fixed "coreset_ratio" key becomes an optional extra_fields: dict
     the caller supplies (omitting it is fine — no error, just no extra key)
Both changes are backward compatible (new parameter has a default) and
don't break existing call sites if this file were reused as-is elsewhere —
worth backporting to PCB-Anomaly-Baselines-PatchCore too for system-wide
consistency (see README, "Differences from PCB-Anomaly-Baselines-PatchCore").

SAVE_PATH / OUTPUT_PATH split:
  SAVE_PATH   — numeric/log: .npz, .json, .csv  (reloadable for further computation)
  OUTPUT_PATH — images: .png only               (for visual inspection only)
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve


# ── SAVE_PATH artifacts (ตัวเลข/log) ─────────────────────────────────────

def save_final_results(cfg, split_name: str, metrics: dict,
                        threshold: float,
                        naive_baselines: dict = None,
                        extra_fields: dict = None) -> Path:
    """เซฟ final_results_{split_name}.json ลง SAVE_PATH — รูปแบบเดียวกับ
    repo หลัก (config snapshot + metrics + naive_baselines) เพื่อให้ script
    เทียบผลข้าม repo โหลด key เดียวกันได้โดยไม่ต้องแก้ไขอะไรเพิ่ม

    naive_baselines: dict จาก compute_naive_baseline_metrics() —
    เก็บ 3 baseline (always_normal, always_anomaly, random_prior) พร้อม
    seed เพื่อให้ผล random_prior reproduce ได้ข้าม run

    extra_fields: dict เสริมของ method-specific hyperparameter (เช่น
    DFR: {"dfr_epochs": cfg.DFR_EPOCHS, "latent_dim": actual_latent_dim})
    ถูก merge เข้า top-level ของ JSON ตรงๆ — ไม่บังคับส่ง

    final_results_{split_name}.json goes to SAVE_PATH — same format as the
    main repo (config snapshot + metrics + naive_baselines) so cross-repo
    comparison scripts load the same keys without modification.
    """
    out = {
        "experiment"           : cfg.EXPERIMENT,
        "backbone"             : cfg.BACKBONE,
        "method"               : cfg.METHOD_NAME,
        "split"                : split_name,
        "threshold"            : threshold,
        "threshold_percentile" : cfg.THRESHOLD_PERCENTILE,
        # กรอง key ที่เป็น array ขนาดใหญ่ออก เพราะ JSON ไม่รองรับ ndarray
        # และ key เหล่านี้อยู่ใน scores_{split}.npz แล้ว
        # Filter out large-array keys (JSON doesn't support ndarray;
        # those already live in scores_{split}.npz).
        "metrics" : {k: (v.tolist() if hasattr(v, 'tolist') else v)
                     for k, v in metrics.items()
                     if k not in ("cm", "fpr", "tpr", "gt", "pred", "scores")},
        "confusion_matrix" : metrics["cm"].tolist(),
    }
    if extra_fields:
        out.update(extra_fields)
    if naive_baselines is not None:
        out["naive_baselines"] = naive_baselines
    out_path = Path(cfg.SAVE_PATH) / f"final_results_{split_name}.json"
    out_path.write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def save_scores(cfg, split_name: str,
                scores: np.ndarray, y_true: np.ndarray,
                labels: list, paths: list,
                heatmaps: np.ndarray,
                orig_imgs: np.ndarray,
                preproc_imgs: np.ndarray) -> Path:
    """เซฟ scores_{split_name}.npz ลง SAVE_PATH ด้วย schema เดียวกับ repo
    หลัก (7 key) — visualize.py และ script เทียบผลข้าม repo โหลด key
    เดียวกันได้ทันทีโดยไม่ต้องแก้ไขอะไร

    key ที่บันทึก:
      scores       float32 [N]        : anomaly score ต่อภาพ
      y_true       int64   [N]        : 0=normal 1=anomaly (ใช้คำนวณ metric)
      labels       string  [N]        : "good"/"defect" (ใช้ display เท่านั้น)
      paths        string  [N]        : path ต้นฉบับ
      heatmaps     float32 [N,H,W]   : regional reconstruction-error map
                                        หลัง upsample+smooth
      orig_imgs    float32 [N,H,W,3] : RGB ต้นฉบับก่อน normalize
      preproc_imgs float32 [N,H,W,3] : ภาพหลัง preprocessing จริง

    Saves scores_{split_name}.npz to SAVE_PATH with the same 7-key schema
    as the main repo — visualize.py and cross-repo scripts load the same
    keys immediately without modification.
    """
    out_path = Path(cfg.SAVE_PATH) / f"scores_{split_name}.npz"
    np.savez_compressed(
        out_path,
        scores       = scores.astype(np.float32),
        y_true       = y_true.astype(np.int64),
        labels       = np.array(labels),
        paths        = np.array(paths),
        heatmaps     = heatmaps.astype(np.float32),
        orig_imgs    = orig_imgs.astype(np.float32),
        preproc_imgs = preproc_imgs.astype(np.float32),
    )
    return out_path


def save_roc_csv(cfg, split_name: str,
                  scores: np.ndarray, y_true: np.ndarray) -> Path:
    """เซฟ roc_curve_data_{split_name}.csv ลง SAVE_PATH — จุดดิบทุกจุดบน
    ROC curve (fpr, tpr, threshold) สำหรับ:
      - เปิดดูใน Excel โดยไม่ต้องรัน Python เพิ่ม
      - multi-seed ROC aggregation (vertical averaging) — โหลด fpr/tpr
        ของแต่ละ seed มา np.interp() เข้าแกน FPR ร่วมก่อน average

    Save roc_curve_data_{split_name}.csv to SAVE_PATH — raw ROC curve
    points (fpr, tpr, threshold) for:
      - Direct viewing in Excel without any extra Python
      - Multi-seed ROC aggregation (vertical averaging): load each seed's
        fpr/tpr and np.interp() onto a shared FPR axis before averaging
    """
    fpr, tpr, thr = roc_curve(y_true, scores)
    out_path = Path(cfg.SAVE_PATH) / f"roc_curve_data_{split_name}.csv"
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("fpr,tpr,threshold\n")
        for a, b, c in zip(fpr, tpr, thr):
            f.write(f"{a},{b},{c}\n")
    return out_path


def compute_heatmap_calibration(
    pixel_maps : np.ndarray,   # [N, H, W]
    y_true     : np.ndarray,   # [N]
    low_pct    : float = 1.0,
    high_pct   : float = 99.5,
) -> "tuple[float, float]":
    """คำนวณ (vmin, vmax) สำหรับ normalize heatmap แบบ global จาก percentile
    ของ pixel_maps บนภาพ normal (y_true==0) เท่านั้น — หลักการเดียวกับ
    select_percentile_threshold() แต่ทำกับ pixel-level distance แทน
    image-level score

    จำเป็นเพราะ pixel_maps จาก DFR.score() เป็น raw L2 reconstruction-error
    map (ไม่เคย normalize มาก่อน) ในขณะที่ src/visual.py (overlay_heatmap)
    สมมติว่า input อยู่ในช่วง [0,1] แล้วเสมอ (heat.clip(0,1)) — ถ้าไม่
    normalize ก่อน แทบทุก pixel จะโดน clip เป็น 1.0 ทำให้ heatmap ออกมา
    แดงเต็มภาพแทบทุกใบ ไม่มี contrast เหลือเลย (บั๊กเดียวกับที่เจอใน
    PatchCore ตอน sync ครั้งก่อน — DFR error map มีปัญหาสเกลแบบเดียวกัน
    เพราะเป็น raw L2 distance เหมือนกัน)

    Compute a global (vmin, vmax) for heatmap normalization from the
    percentile of pixel_maps on normal images (y_true==0) only — same
    principle as select_percentile_threshold(), applied to pixel-level
    distance instead of the image-level score.

    Required because DFR.score()'s pixel_maps are raw L2 reconstruction-
    error maps (never normalized), while src/visual.py's overlay_heatmap()
    always assumes [0,1] input (heat.clip(0,1)) — without normalizing
    first, almost every pixel gets clipped to 1.0, producing a heatmap
    that's solid red almost everywhere with zero remaining contrast (the
    same bug class found in PatchCore during a previous sync — DFR's error
    map has the same scaling issue since it's also a raw L2 distance).
    """
    normal_maps = pixel_maps[y_true == 0]
    if normal_maps.shape[0] == 0:
        raise ValueError(
            'compute_heatmap_calibration() ต้องการภาพ normal อย่างน้อย 1 '
            'ภาพ (y_true==0) แต่ไม่พบเลยใน pixel_maps/y_true ที่ส่งเข้ามา'
        )
    vmin = float(np.percentile(normal_maps, low_pct))
    vmax = float(np.percentile(normal_maps, high_pct))
    if vmax <= vmin:
        vmax = vmin + 1e-8
    return vmin, vmax


def normalize_pixel_maps_global(
    pixel_maps : np.ndarray,   # [N, H, W]
    vmin       : float,
    vmax       : float,
) -> np.ndarray:
    """Normalize pixel_maps ทั้งก้อนด้วย (vmin, vmax) เดียวกัน (global,
    ไม่ใช่ per-image) แล้ว clip เข้า [0,1] — เรียกก่อน save_scores() เสมอ
    เพื่อให้ heatmap ที่เซฟไว้ใช้กับ visualize.py ได้ถูกต้อง

    Normalize the whole pixel_maps array with the same (vmin, vmax) —
    global, not per-image — then clip to [0,1]. Always call this before
    save_scores() so the saved heatmap works correctly with visualize.py.
    """
    return np.clip((pixel_maps - vmin) / (vmax - vmin), 0.0, 1.0).astype(np.float32)


def save_training_history(cfg, history: dict) -> Path:
    """เซฟ history.json ลง SAVE_PATH — ต่างจาก PatchCore/PaDiM ตรงที่ DFR
    มี trainable CAE จริง จึงมี training history ให้เซฟ (PatchCore/PaDiM
    ไม่มีฟังก์ชันนี้เพราะไม่มี training loop เลย)

    history ต้องมีอย่างน้อย key 'train_loss' (list[float] ต่อ epoch) —
    ดู README หัวข้อ "ความต่างจาก repo หลัก" ว่าทำไมมีแค่ train_loss
    (ไม่มี val_loss/val_auroc เหมือน EXPERIMENT 0) เพราะ fit() ตาม
    BaseAnomalyModel ไม่ได้รับ val_loader มาด้วย

    Saves history.json to SAVE_PATH — unlike PatchCore/PaDiM (no training
    loop, so this function doesn't exist for them), DFR has a real
    trainable CAE and therefore a training history worth saving.
    """
    out_path = Path(cfg.SAVE_PATH) / "history.json"
    out_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return out_path


def write_save_path_readme(cfg) -> Path:
    """สร้าง README.md ใน SAVE_PATH อธิบายไฟล์ทุกตัวที่มีอยู่จริง ณ ตอนเรียก
    (dynamic — list เฉพาะไฟล์ที่ exist) เพื่อให้คนที่เปิดโฟลเดอร์ทราบว่า
    แต่ละไฟล์เก็บอะไรและเอาไปทำอะไรได้บ้าง

    Writes README.md into SAVE_PATH describing every file that actually
    exists at call time (dynamic — only lists files that exist), so anyone
    opening the folder knows what each file holds and what it can be used for.
    """
    from datetime import datetime

    p = Path(cfg.SAVE_PATH)
    method = getattr(cfg, "METHOD_NAME", "DFR")
    lines = [
        f"# Artifacts ใน SAVE_PATH — {method}\n",
        f"สร้างอัตโนมัติเมื่อ {datetime.now().isoformat(timespec='seconds')}\n\n",
        "โฟลเดอร์นี้เก็บ **ตัวเลข/log ทั้งหมด** — ไม่มีไฟล์ภาพ\n",
        f"(ภาพอยู่ที่ `{cfg.OUTPUT_PATH}`)\n\n---\n\n",
    ]

    docs = {
        "history.json": (
            "training history ของ CAE: {'train_loss': [float, ...]} หนึ่งค่าต่อ epoch "
            "(normal-only, ไม่มี val monitoring — ดู README)"
        ),
        "scores_val.npz": (
            "7-key array: scores(float32), y_true(int64), labels(str), paths(str), "
            "heatmaps(float32,[N,H,W]), orig_imgs(float32,[N,H,W,3]), "
            "preproc_imgs(float32,[N,H,W,3]) — โหลดด้วย np.load(..., allow_pickle=True)"
        ),
        "scores_test.npz": "เหมือน scores_val.npz แต่สำหรับ test split",
        "final_results_val.json": (
            "config snapshot + metrics ครบชุด (val) — "
            "เทียบ AUROC/escape_rate กับ repo อื่นได้โดยตรง"
        ),
        "final_results_test.json": "เหมือน final_results_val.json แต่สำหรับ test split",
        "roc_curve_data_val.csv": (
            "fpr, tpr, threshold ทุกจุดบน ROC curve (val) — "
            "เปิดใน Excel ได้เลย หรือใช้เป็น input ของ multi-seed ROC aggregation"
        ),
        "roc_curve_data_test.csv": "เหมือน roc_curve_data_val.csv แต่สำหรับ test split",
        "cost_aware_sweep.csv": "cost-aware threshold sweep (ถ้าเคยรัน scripts/run_cost_aware_dfr.py)",
        "README.md": "ไฟล์นี้ — auto-generated",
    }

    for fname, desc in docs.items():
        fpath = p / fname
        if fpath.exists():
            size = fpath.stat().st_size
            size_str = (f"{size/1024:.1f} KB" if size < 1024**2
                        else f"{size/1024**2:.1f} MB")
            lines.append(f"## `{fname}` ({size_str})\n{desc}\n\n---\n\n")

    readme_path = p / "README.md"
    readme_path.write_text("".join(lines), encoding="utf-8")
    return readme_path
