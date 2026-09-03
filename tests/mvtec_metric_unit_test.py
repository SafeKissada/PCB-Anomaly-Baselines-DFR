"""
Unit test สำหรับ src/mvtec_evaluate.py — ตรวจ pixel_roc_auc()/pro_auc()
ด้วย synthetic ground truth ที่รู้คำตอบล่วงหน้า ก่อนเอาไปเชื่อกับ DFR จริง
(ถ้า metric เองคำนวณผิด ต่อให้ DFR ทำงานถูก ตัวเลขที่ได้ก็ผิดอยู่ดี)

Usage: python tests/mvtec_metric_unit_test.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mvtec_evaluate import pixel_roc_auc, pro_auc


def test_perfect_prediction():
    """pixel_map == mask เป๊ะ (scaled 0/1) -> ROC-AUC และ PRO-AUC ต้องเป็น
    1.0 ทั้งคู่ (แยก normal/anomaly ได้สมบูรณ์แบบ, region overlap 100%)
    """
    rng = np.random.RandomState(0)
    N, H, W = 20, 32, 32
    masks = np.zeros((N, H, W), dtype=np.uint8)
    for i in range(0, N, 2):  # ครึ่งหนึ่งเป็นภาพ anomaly มี region สี่เหลี่ยม
        y0, x0 = rng.randint(0, H - 8), rng.randint(0, W - 8)
        masks[i, y0:y0 + 8, x0:x0 + 8] = 1

    pixel_maps = masks.astype(np.float64)  # เหมือนเป๊ะ

    roc = pixel_roc_auc(masks, pixel_maps)
    pro = pro_auc(masks, pixel_maps, max_fpr=0.30, num_thresholds=100)
    print(f"[perfect]      ROC-AUC={roc:.4f}  PRO-AUC={pro:.4f}")
    assert roc > 0.999, f"Expected ROC-AUC ~1.0 for perfect prediction, got {roc}"
    assert pro > 0.999, f"Expected PRO-AUC ~1.0 for perfect prediction, got {pro}"


def test_random_prediction():
    """pixel_map สุ่มล้วนๆ ไม่เกี่ยวกับ mask เลย -> ROC-AUC ต้องใกล้ 0.5
    (chance level) — PRO-AUC ไม่มี "chance=0.5" ที่ตรงตัวขนาดนั้น แต่ต้อง
    ต่ำกว่า perfect-prediction case อย่างชัดเจน
    """
    rng = np.random.RandomState(1)
    N, H, W = 20, 32, 32
    masks = np.zeros((N, H, W), dtype=np.uint8)
    for i in range(0, N, 2):
        y0, x0 = rng.randint(0, H - 8), rng.randint(0, W - 8)
        masks[i, y0:y0 + 8, x0:x0 + 8] = 1

    pixel_maps = rng.rand(N, H, W)  # สุ่มล้วนๆ ไม่เกี่ยวกับ masks

    roc = pixel_roc_auc(masks, pixel_maps)
    pro = pro_auc(masks, pixel_maps, max_fpr=0.30, num_thresholds=100)
    print(f"[random]       ROC-AUC={roc:.4f}  PRO-AUC={pro:.4f}")
    assert 0.35 < roc < 0.65, f"Expected ROC-AUC ~0.5 for random prediction, got {roc}"
    assert pro < 0.5, f"Expected low PRO-AUC for random prediction, got {pro}"


def test_all_normal_no_regions():
    """ไม่มี ground-truth region เลย (mask ทุกภาพ all-zero) -> ทั้งคู่ต้อง
    คืน NaN อย่างสุภาพ ไม่ crash (สถานการณ์นี้เกิดได้จริงถ้า test set มีแต่
    ภาพ normal ล้วน — ไม่ควรเกิดกับ MVTec ปกติเพราะทุก category มีภาพ
    anomaly อยู่แล้ว แต่ต้องกันไว้เผื่อ mask หายหมดเพราะบั๊กอื่น)
    """
    N, H, W = 10, 16, 16
    masks = np.zeros((N, H, W), dtype=np.uint8)
    pixel_maps = np.random.rand(N, H, W)

    roc = pixel_roc_auc(masks, pixel_maps)
    pro = pro_auc(masks, pixel_maps)
    print(f"[all-normal]   ROC-AUC={roc}  PRO-AUC={pro}")
    assert np.isnan(roc), f"Expected NaN when masks have no positive pixels, got {roc}"
    assert np.isnan(pro), f"Expected NaN when masks have no regions, got {pro}"


def test_partial_overlap():
    """prediction ครอบคลุมแค่ครึ่งเดียวของแต่ละ region -> PRO ต้องอยู่
    แถวๆ 0.5 ไม่ใช่ 0 หรือ 1 (เช็คว่าคำนวณ overlap ratio ถูกจริง ไม่ใช่แค่
    binary hit/miss)
    """
    N, H, W = 10, 32, 32
    masks = np.zeros((N, H, W), dtype=np.uint8)
    pixel_maps = np.zeros((N, H, W), dtype=np.float64)
    for i in range(0, N, 2):
        masks[i, 10:20, 10:20] = 1          # region 10x10
        pixel_maps[i, 10:15, 10:20] = 1.0   # prediction ครอบแค่ครึ่งบน (10x5)
    # ภาพ normal (i คี่): pixel_maps เป็น 0 หมด (ไม่ false-positive เลย)

    pro = pro_auc(masks, pixel_maps, max_fpr=1.0, num_thresholds=50)
    print(f"[half-overlap] PRO-AUC={pro:.4f}")
    # threshold ต่ำสุด (>=0) จะ flag ทุก pixel เป็น anomaly (pred=all-True
    # เพราะ pixel_maps>=0 จริงเสมอ) ทำให้ overlap=100% ที่ threshold นั้น —
    # แต่ threshold สูงกว่า 0 (>=1.0) จะได้ overlap=50% พอดี เฉลี่ยทั้ง
    # curve ควรอยู่ระหว่าง 0.5-1.0 ไม่ใช่สุดขั้วไปทางใดทางหนึ่ง
    assert 0.3 < pro < 1.0, f"Expected partial PRO-AUC in a sane middle range, got {pro}"


if __name__ == "__main__":
    test_perfect_prediction()
    test_random_prediction()
    test_all_normal_no_regions()
    test_partial_overlap()
    print("\n✅ ALL MVTEC METRIC UNIT TESTS PASSED")
