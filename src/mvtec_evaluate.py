"""
Pixel-level ROC-AUC และ PRO-AUC — metric ที่ paper รายงานจริงใน TABLE II/III
(ต่างจาก src/evaluate.py ที่คำนวณ classification metric ระดับ**ภาพ**
scalar score เดียวต่อภาพ — ที่นี่ทำงานบน pixel-level anomaly map เต็มภาพ
เทียบกับ pixel-level ground-truth mask)

- **ROC-AUC**: "assesses the best potential segmentation result in terms
  of normal and anomalous pixels, i.e. per pixel overlapping performance"
  (paper section IV-A.5) — sklearn roc_auc_score ธรรมดา แต่ flatten รวม
  พิกเซลทุกภาพ (ทั้ง normal และ anomaly) เข้าด้วยกันก่อน ไม่ใช่เฉลี่ยต่อภาพ

- **PRO-AUC** (Per-Region-Overlap): เสนอใน Bergmann et al. 2019 (MVTec AD
  benchmark paper, citation [7] ใน DFR paper) วัดว่า model segment ได้ครบ
  ทุก ground-truth region เท่าๆ กันแค่ไหน (ไม่ bias ไปทาง region ใหญ่แบบที่
  pixel-level metric ธรรมดาทำ) — DFR paper ใช้ normalized PRO-AUC จำกัดที่
  average pixel-FPR ไม่เกิน 30% (section IV-A.5, "we report the normalized
  PRO-AUC up to an average per-pixel false positive rate (FPR) of 30%")

อัลกอริทึม PRO-AUC (matching Bergmann et al. 2019's original protocol,
also used by anomalib's PRO metric):
  1. สำหรับแต่ละ threshold (sweep จากมากไปน้อย): binarize anomaly map
  2. คำนวณ pixel-level FPR จากพิกเซล "ปกติ" ทั้งหมด (ทุกพิกเซลที่
     ground-truth=0 ไม่ว่าจะอยู่ในภาพ normal หรือภาพ anomaly ก็ตาม)
  3. สำหรับแต่ละ ground-truth region (connected component ของ mask แต่ละ
     ภาพ anomaly): คำนวณ overlap ratio = |prediction ∩ region| / |region|
  4. เฉลี่ย overlap ratio ข้ามทุก region ทุกภาพที่ threshold นั้น = PRO score
  5. ได้ curve (FPR, PRO) ตาม threshold ที่ sweep มา ตัดที่ FPR <= max_fpr
     แล้ว trapezoidal-integrate หารด้วย max_fpr = normalized PRO-AUC
"""
import logging
from typing import Optional

import numpy as np
from sklearn.metrics import roc_auc_score
from skimage.measure import label as cc_label

logger = logging.getLogger("mvtec_evaluate")


def pixel_roc_auc(masks: np.ndarray, pixel_maps: np.ndarray) -> float:
    """ROC-AUC ระดับ pixel รวมทุกภาพ (ทั้ง normal และ anomaly) เข้าด้วยกัน
    ก่อน flatten — ตรงตาม paper section IV-A.5

    masks, pixel_maps: [N, H, W] — masks เป็น {0,1}, pixel_maps เป็น
    continuous anomaly score (ไม่ต้อง normalize มาก่อน, ROC-AUC เป็น
    ranking metric ไม่สนใจ scale)
    """
    y = masks.reshape(-1)
    s = pixel_maps.reshape(-1)
    if len(np.unique(y)) < 2:
        logger.warning("pixel_roc_auc(): mask ทั้งชุดมีแต่ค่าเดียว (all-0 "
                       "หรือ all-1) — ROC-AUC ไม่มีความหมาย คืน NaN")
        return float("nan")
    return float(roc_auc_score(y, s))


def pro_auc(masks: np.ndarray, pixel_maps: np.ndarray,
           max_fpr: float = 0.30, num_thresholds: int = 200) -> float:
    """Normalized PRO-AUC จำกัดที่ average pixel-FPR ไม่เกิน max_fpr — ดู
    docstring หัวไฟล์สำหรับอัลกอริทึมเต็ม

    masks, pixel_maps: [N, H, W] เหมือน pixel_roc_auc()

    O(num_thresholds × total_regions) — เหมาะกับขนาด test set ต่อ category
    ของ MVTec AD ปกติ (~40-150 ภาพ) ถ้าข้อมูลใหญ่กว่านี้มากให้ลด
    num_thresholds ลง (แลก resolution ของ curve กับความเร็ว)
    """
    if masks.shape != pixel_maps.shape:
        raise ValueError(
            f"pro_auc(): masks.shape={masks.shape} != "
            f"pixel_maps.shape={pixel_maps.shape} — ต้อง resize ให้เท่ากัน "
            f"ก่อนเรียกฟังก์ชันนี้เสมอ (ดู src/data/mvtec_dataset.py::load_mask)")

    # ── precompute: connected-component region ของแต่ละภาพ (ทำครั้งเดียว) ──
    regions_per_image = []
    total_regions = 0
    for m in masks:
        if m.sum() == 0:
            regions_per_image.append([])
            continue
        lbl = cc_label(m.astype(np.int32))
        n_regions = int(lbl.max())
        regions = [(lbl == r) for r in range(1, n_regions + 1)]
        regions_per_image.append(regions)
        total_regions += n_regions

    if total_regions == 0:
        logger.warning("pro_auc(): ไม่มี ground-truth region เลยสักภาพ "
                       "(mask ทุกภาพเป็น all-zero) — PRO-AUC ไม่มีความหมาย "
                       "คืน NaN")
        return float("nan")

    normal_pixel_mask = (masks == 0)
    n_normal_pixels = int(normal_pixel_mask.sum())
    if n_normal_pixels == 0:
        logger.warning("pro_auc(): ไม่มีพิกเซล 'ปกติ' เลย (mask ทุกภาพเป็น "
                       "all-one) — คำนวณ FPR ไม่ได้ คืน NaN")
        return float("nan")

    # ── threshold sweep (มาก -> น้อย, ให้ FPR ไล่จากต่ำไปสูงเป็นธรรมชาติ) ──
    thresholds = np.linspace(pixel_maps.max(), pixel_maps.min(), num_thresholds)

    fprs = np.empty(num_thresholds, dtype=np.float64)
    pros = np.empty(num_thresholds, dtype=np.float64)

    for t_idx, t in enumerate(thresholds):
        pred = pixel_maps >= t  # [N, H, W] bool

        fp = np.logical_and(pred, normal_pixel_mask).sum()
        fprs[t_idx] = fp / n_normal_pixels

        pro_sum, pro_count = 0.0, 0
        for img_idx, regions in enumerate(regions_per_image):
            if not regions:
                continue
            pred_i = pred[img_idx]
            for region in regions:
                overlap = np.logical_and(pred_i, region).sum()
                pro_sum += overlap / region.sum()
                pro_count += 1
        pros[t_idx] = pro_sum / pro_count if pro_count > 0 else 0.0

    # ── restrict to FPR <= max_fpr แล้ว normalize-integrate ──────────
    order = np.argsort(fprs)
    fprs_sorted, pros_sorted = fprs[order], pros[order]

    keep = fprs_sorted <= max_fpr
    if keep.sum() < 2:
        logger.warning(
            f"pro_auc(): มีจุดข้อมูลแค่ {keep.sum()} จุดที่ FPR<={max_fpr} "
            f"(น้อยเกินไปจะ integrate) — เพิ่ม num_thresholds หรือเช็คว่า "
            f"pixel_maps มี dynamic range พอสมควรหรือไม่ คืน NaN")
        return float("nan")

    fprs_r, pros_r = fprs_sorted[keep], pros_sorted[keep]
    # เพิ่มจุด (max_fpr, ค่า pro ที่ threshold ต่ำสุดที่ยัง <= max_fpr) เข้า
    # ปลาย curve ถ้ายังไม่ถึง max_fpr พอดี (linear-interpolate) กัน
    # under-estimate จากการตัด curve ก่อนถึง max_fpr จริง
    if fprs_r[-1] < max_fpr:
        # หาจุดถัดไปที่ FPR > max_fpr เพื่อ interpolate เข้าหา max_fpr พอดี
        beyond = np.where(fprs_sorted > max_fpr)[0]
        if len(beyond) > 0:
            i_next = beyond[0]
            f0, f1 = fprs_sorted[i_next - 1], fprs_sorted[i_next]
            p0, p1 = pros_sorted[i_next - 1], pros_sorted[i_next]
            if f1 > f0:
                p_interp = p0 + (p1 - p0) * (max_fpr - f0) / (f1 - f0)
                fprs_r = np.append(fprs_r, max_fpr)
                pros_r = np.append(pros_r, p_interp)

    auc = _trapz(pros_r, fprs_r) / max_fpr
    return float(np.clip(auc, 0.0, 1.0))


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """np.trapz ถูกเปลี่ยนชื่อเป็น np.trapezoid ใน numpy>=2.0 (trapz ยัง
    เรียกได้ใน numpy 1.x แต่หายไปแล้วใน numpy 2.x บางเวอร์ชัน) — เช็คว่า
    attribute ไหนมีอยู่จริงแล้วเรียกอันนั้น กันพังข้าม numpy version
    """
    fn = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(fn(y, x))
