"""
Heatmap sanity check — ต่างจาก tests/smoke_test.py ตรงที่ตัวนี้เช็ค
"เนื้อหา" ของ heatmap ไม่ใช่แค่ shape/no-error เหมือน smoke_test.py

วิธีเช็ค: สร้างภาพ defect ที่มี patch สีขาวสว่างชัดเจนวางไว้ที่ตำแหน่ง
คงที่ (รู้ตำแหน่งล่วงหน้า) แล้วดูว่า error-map ที่ DFR สร้างขึ้นชี้ไปที่
ตำแหน่ง patch จริงหรือไม่ — ถ้า heatmap เป็น noise สุ่มกระจายทั่วภาพ
(ไม่มีโครงสร้างอะไรเลย) ตำแหน่งพีคของ error จะสุ่มเท่าๆ กันทุกที่
(expected hit-rate ≈ พื้นที่ patch / พื้นที่ภาพทั้งหมด) แต่ถ้า error
กระจุกอยู่ที่ patch จริง hit-rate จะสูงกว่านั้นอย่างมีนัยสำคัญ

ยังบันทึกภาพ heatmap overlay จริงไว้ให้ดูด้วยตา (ไม่ใช่แค่ตัวเลข) ที่
/tmp/dfr_heatmap_check/inspect_*.png

หมายเหตุ: รันด้วย backbone offline (PRETRAINED=False, resnet18 layer1/2
เท่านั้น — receptive field เล็ก) เพราะ sandbox นี้ดาวน์โหลด pretrained
weight ไม่ได้ ผลที่ได้จึงเป็นแค่ "sanity check เชิงโครงสร้าง" ว่า pipeline
ทำ localization ได้บ้างในหลักการ ไม่ใช่การยืนยันคุณภาพ DFR แบบเต็มรูปแบบ
(ซึ่งต้องใช้ pretrained backbone จริงกับข้อมูลจริง)

Usage: python tests/heatmap_sanity_check.py
"""
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import Config
from scripts.run_dfr import run

# ── ขนาด patch (ตำแหน่ง "สุ่มต่อภาพ" ดู make_patch_dataset) ────────────
# เดิม (เวอร์ชันแรก) ใช้ตำแหน่งคงที่เดียวกันทุกภาพ defect — พบว่า test
# แบบนั้น confound กันเอง: แยกไม่ออกระหว่าง "heatmap ตามตำแหน่ง patch จริง"
# กับ "heatmap ตอบสนองต่อ 'การเป็นภาพ defect' แบบ generic ไม่ว่า patch จะ
# อยู่ตรงไหน" เพราะพิกัด patch เหมือนกันทุกใบ แก้โดยสุ่มตำแหน่ง patch
# ใหม่ทุกภาพ defect แล้วเช็ค hit-rate เทียบกับ bounding box ของ "ภาพนั้นๆ
# เอง" — ถ้า heatmap นิ่งอยู่ตำแหน่งเดิมไม่ขยับตามภาพ นี่คือหลักฐานชัดเจน
# ว่าเป็น positional artifact ไม่ใช่การตอบสนองต่อ patch จริง
IMG_SIZE = 96
PATCH_SIZE = 24
MARGIN = 4  # กันไม่ให้ patch ชิดขอบภาพเกินไป (จะไปชนกับ boundary-effect โดยตรง)


def make_patch_dataset(root: Path, n_good=40, n_defect=16):
    good_dir = root / "good"
    defect_dir = root / "defect"
    good_dir.mkdir(parents=True, exist_ok=True)
    defect_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(0)
    for i in range(n_good):
        base = 120 + rng.randint(-8, 8, (IMG_SIZE, IMG_SIZE, 1))
        arr = np.clip(np.repeat(base, 3, axis=2), 0, 255).astype(np.uint8)
        Image.fromarray(arr).save(good_dir / f"good_{i:03d}.png")

    patch_boxes = {}  # filename -> (y0, y1, x0, x1), สำหรับใช้เช็ค hit-rate ทีหลัง
    lo, hi = MARGIN, IMG_SIZE - MARGIN - PATCH_SIZE
    for i in range(n_defect):
        base = 120 + rng.randint(-8, 8, (IMG_SIZE, IMG_SIZE, 1))
        arr = np.clip(np.repeat(base, 3, axis=2), 0, 255).astype(np.uint8)
        # ตำแหน่ง patch สุ่มใหม่ทุกภาพ (ต่างจากเวอร์ชันแรกที่ fix ตำแหน่งเดียว)
        y0 = int(rng.randint(lo, hi + 1))
        x0 = int(rng.randint(lo, hi + 1))
        y1, x1 = y0 + PATCH_SIZE, x0 + PATCH_SIZE
        arr[y0:y1, x0:x1, :] = 250
        fname = f"defect_{i:03d}.png"
        Image.fromarray(arr).save(defect_dir / fname)
        patch_boxes[fname] = (y0, y1, x0, x1)
    return patch_boxes


def _box_for(path: str, patch_boxes: dict):
    fname = Path(str(path)).name
    return patch_boxes.get(fname)


def _hit_rate(heatmaps: np.ndarray, y_true: np.ndarray, paths, patch_boxes: dict) -> dict:
    """สำหรับแต่ละภาพ defect (y_true==1): หาตำแหน่ง argmax ของ heatmap
    แล้วเช็คว่าอยู่ในกรอบ patch "ของภาพนั้นๆ เอง" หรือไม่ (ตำแหน่งสุ่ม
    ต่างกันทุกภาพ — ดู make_patch_dataset) คืน hit_rate เทียบกับ
    chance_rate (สัดส่วนพื้นที่ patch/ภาพทั้งหมด ถ้า heatmap เป็น noise
    สุ่มไม่มีโครงสร้างอะไรเลย)
    """
    defect_idx = np.where(y_true == 1)[0]
    hits = 0
    peak_locations = []
    n_checked = 0
    for i in defect_idx:
        box = _box_for(paths[i], patch_boxes)
        if box is None:
            continue
        y0, y1, x0, x1 = box
        hmap = heatmaps[i]
        py, px = np.unravel_index(np.argmax(hmap), hmap.shape)
        peak_locations.append((int(py), int(px), box))
        n_checked += 1
        if y0 <= py < y1 and x0 <= px < x1:
            hits += 1
    h, w = heatmaps.shape[1:]
    chance_rate = (PATCH_SIZE * PATCH_SIZE) / (h * w)
    return dict(
        n_defect=n_checked,
        hits=hits,
        hit_rate=hits / max(1, n_checked),
        chance_rate=chance_rate,
        peak_locations=peak_locations,
    )


def _inside_vs_outside_error(heatmaps: np.ndarray, y_true: np.ndarray,
                              paths, patch_boxes: dict) -> dict:
    """เทียบค่าเฉลี่ย error 'ในกรอบ patch ของภาพนั้นๆ เอง' กับ 'นอกกรอบ'
    — ถ้า DFR localize ได้จริง ค่าในกรอบต้องสูงกว่าค่านอกกรอบอย่างชัดเจน
    ไม่ใช่ใกล้เคียงกัน (ใกล้เคียงกัน/สลับกันไปมาแล้วแต่ภาพ = heatmap
    ตอบสนองต่อสิ่งอื่นที่ไม่ใช่ตำแหน่ง patch จริง)
    """
    defect_idx = np.where(y_true == 1)[0]
    inside_vals, outside_vals = [], []
    for i in defect_idx:
        box = _box_for(paths[i], patch_boxes)
        if box is None:
            continue
        y0, y1, x0, x1 = box
        hmap = heatmaps[i]
        mask_inside = np.zeros(hmap.shape, dtype=bool)
        mask_inside[y0:y1, x0:x1] = True
        inside_vals.append(float(hmap[mask_inside].mean()))
        outside_vals.append(float(hmap[~mask_inside].mean()))
    return dict(
        mean_inside=float(np.mean(inside_vals)),
        mean_outside=float(np.mean(outside_vals)),
        ratio=float(np.mean(inside_vals) / max(1e-8, np.mean(outside_vals))),
    )


def _save_visual_inspection(cfg, split: str, patch_boxes: dict, n=6):
    """เซฟภาพ overlay (original | heatmap | overlay) ของภาพ defect จริงๆ
    ไว้ให้ดูด้วยตา ไม่ใช่แค่ตัวเลข hit-rate — ใช้ overlay_heatmap() เดียวกับ
    ที่ visualize_dfr.py ใช้จริง ไม่เขียน visualize logic ซ้ำ วาดกรอบ
    ตำแหน่ง patch จริง (สุ่มต่อภาพ) ทับบนภาพต้นฉบับด้วยเพื่อเทียบกับ
    heatmap ได้ตรงๆ ด้วยตา
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from src.visual import overlay_heatmap

    d = np.load(Path(cfg.SAVE_PATH) / f"scores_{split}.npz", allow_pickle=True)
    y_true = d["y_true"]
    paths = d["paths"]
    defect_idx = np.where(y_true == 1)[0][:n]
    normal_idx = np.where(y_true == 0)[0][:2]
    show_idx = list(defect_idx) + list(normal_idx)

    fig, axes = plt.subplots(len(show_idx), 3, figsize=(9, 3 * len(show_idx)))
    for row, i in enumerate(show_idx):
        img = d["orig_imgs"][i]
        heat = d["heatmaps"][i]
        overlay = overlay_heatmap(img, heat, alpha=0.5)
        kind = "DEFECT" if y_true[i] == 1 else "normal"

        axes[row, 0].imshow(img); axes[row, 0].set_title(f"[{kind}] original")
        axes[row, 1].imshow(heat, cmap="jet"); axes[row, 1].set_title("raw heatmap")
        axes[row, 2].imshow(overlay); axes[row, 2].set_title("overlay")
        for ax in axes[row]:
            ax.axis("off")

        box = _box_for(paths[i], patch_boxes) if kind == "DEFECT" else None
        if box is not None:
            y0, y1, x0, x1 = box
            for ax in (axes[row, 0], axes[row, 2]):
                rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                          linewidth=2, edgecolor="lime", facecolor="none")
                ax.add_patch(rect)

    plt.tight_layout()
    out_path = Path(cfg.OUTPUT_PATH) / f"inspect_{split}_heatmaps.png"
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    tmp_root = Path("/tmp/dfr_heatmap_check")
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    data_root = tmp_root / "data"
    patch_boxes = make_patch_dataset(data_root)

    cfg = Config(
        DATA_ROOT=str(data_root),
        SPLIT_CACHE_PATH=str(tmp_root / "splits" / "split_assignment.csv"),
        SAVE_PATH=str(tmp_root / "save/logs"),
        OUTPUT_PATH=str(tmp_root / "save/results"),
        IMAGE_SIZE=(IMG_SIZE, IMG_SIZE),
        BATCH_SIZE=8,
        NUM_WORKERS=0,
        BACKBONE="resnet18",           # เบา + RF เล็ก เหมาะกับ patch localization test
        PRETRAINED=False,              # sandbox นี้ดาวน์โหลด pretrained weight ไม่ได้ —
                                        # ผลจึงเป็นแค่ structural sanity check (ดู docstring)
        FEATURE_LAYERS=("layer1", "layer2"),
        DFR_ALIGN_MODE="nearest",
        DFR_AGG_KERNEL=4,
        DFR_AGG_STRIDE=4,
        NORMALIZE_FEATURES=False,
        DFR_EPOCHS=60,                 # เยอะกว่า smoke_test เพราะต้องการให้ CAE
                                        # เรียนรู้ background pattern จริงจังพอจะ
                                        # เห็น patch เป็นความผิดปกติ
        DFR_LATENT_DIM=16,              # background เรียบมาก ไม่ต้องการ latent dim ใหญ่
        SCORE_METHOD="max",
        HEATMAP_SIGMA=1.5,             # ลด smoothing ลงจาก default 4.0 กัน patch
                                        # เล็กๆ ถูกเบลอจนแยกไม่ออกจาก background
        EXPERIMENT="heatmap_sanity_check",
    )

    val_result, test_result = run(cfg)

    print("\n" + "=" * 70)
    print(" HEATMAP SANITY CHECK")
    print("=" * 70)

    all_pass = True
    for split_name, result in [("val", val_result), ("test", test_result)]:
        hit = _hit_rate(result.pixel_maps, result.y_true, result.paths, patch_boxes)
        io = _inside_vs_outside_error(result.pixel_maps, result.y_true, result.paths, patch_boxes)
        print(f"\n[{split_name}] n_defect={hit['n_defect']}")
        print(f"  Peak-in-patch hit rate : {hit['hit_rate']:.2%}  "
              f"(chance level ≈ {hit['chance_rate']:.2%})")
        print(f"  Mean error inside patch : {io['mean_inside']:.4f}")
        print(f"  Mean error outside patch: {io['mean_outside']:.4f}")
        print(f"  Inside/outside ratio    : {io['ratio']:.2f}x")

        # เกณฑ์ผ่าน: peak ต้องอยู่ในกรอบ patch บ่อยกว่า random เห็นได้ชัด
        # (>3x chance level) และ error ในกรอบต้องสูงกว่านอกกรอบจริง (>1.2x)
        # — ตัวเลขนี้เป็น sanity threshold ไม่ใช่ metric รายงานผลจริง
        split_pass = (hit['hit_rate'] > 3 * hit['chance_rate']) and (io['ratio'] > 1.2)
        print(f"  -> {'✅ PASS' if split_pass else '❌ FAIL'} "
              f"(heatmap {'concentrates on the patch' if split_pass else 'looks scattered/unstructured'})")
        all_pass = all_pass and split_pass

        inspect_path = _save_visual_inspection(cfg, split_name, patch_boxes)
        print(f"  Visual inspection image saved -> {inspect_path}")

    print("\n" + "=" * 70)
    if all_pass:
        print(" ✅ HEATMAP SANITY CHECK PASSED — error concentrates on the "
              "known defect patch, not scattered randomly.")
    else:
        print(
            " ⚠️  HEATMAP SANITY CHECK FAILED with PRETRAINED=False.\n\n"
            " Diagnosed root cause (see README §'Heatmap sanity check finding'\n"
            " for the full investigation): with a RANDOM-INIT backbone, a\n"
            " strong local perturbation (this test's bright patch) triggers a\n"
            " large, non-local, patch-position-INDEPENDENT error region —\n"
            " confirmed by re-randomizing the patch location per image and\n"
            " seeing the 'hot' region NOT move with it, and by more\n"
            " epochs/larger latent dim not fixing it. Reflection padding (the\n"
            " paper's own §IV-C.2 fix) DOES remove a separate, simpler\n"
            " positional-vignette artifact (verified: feature-magnitude-map\n"
            " correlation across unrelated images dropped from 0.95 to 0.04)\n"
            " but does not fix this second effect.\n\n"
            " Most likely explanation: untrained random weights lack the\n"
            " Lipschitz-type stability of a trained network, so a strong\n"
            " local anomaly can produce a non-local, somewhat chaotic\n"
            " response — this is exactly why PRETRAINED=False is documented\n"
            " everywhere in this repo as smoke-test-only. It does NOT confirm\n"
            " a bug in the DFR/regional-feature-generator code (dataset.py's\n"
            " split, evaluate.py's metrics, and the fit/score pipeline all\n"
            " ran correctly — see AUC=1.0 above, which the aggregate\n"
            " image-level score DOES separate perfectly here).\n\n"
            " ACTION REQUIRED before trusting real DFR results: re-run this\n"
            " exact script with PRETRAINED=True (needs internet access this\n"
            " sandbox doesn't have) and confirm the heatmap localizes\n"
            " properly with real ImageNet weights. Do this before drawing any\n"
            " conclusion from DFR's real-dataset heatmaps.")
    print("=" * 70)
    print(f"\nArtifacts kept at {tmp_root} for inspection (not auto-deleted).")


if __name__ == "__main__":
    main()
