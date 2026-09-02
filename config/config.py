"""
Config สำหรับ DFR baseline (Yang et al., "DFR: Deep Feature Reconstruction
for Unsupervised Anomaly Segmentation", arXiv:2012.07122)

DATA_ROOT / GOOD_DIRNAME / DEFECT_DIRNAME / SPLIT_RATIOS / SPLIT_CACHE_PATH /
GROUP_ID_REGEX / SEED / VALID_EXT / IMAGE_SIZE / BATCH_SIZE / NUM_WORKERS /
PIN_MEMORY / THRESHOLD_PERCENTILE / HEATMAP_SIGMA / USE_GRAYSCALE* / USE_CLAHE*
ตั้งชื่อ field เหมือนกับ repo หลัก (Anomaly-Detection-THESIS) และ
PCB-Anomaly-Baselines-PatchCore เป๊ะๆ โดยตั้งใจ — เพื่อให้สามารถชี้
SPLIT_CACHE_PATH ไปที่ไฟล์ split เดียวกันได้ (แนะนำอย่างยิ่งให้ทำแบบนี้
ดู README หัวข้อ "การเทียบผลกับ baseline เดิม") และให้ผลลัพธ์จากทุก repo
เทียบกันได้ตรงๆ โดยไม่มี confound เรื่อง train/val/test membership หรือ
นิยาม metric ต่างกัน

ต่างจาก config ของ PatchCore ตรงที่ตัด field เฉพาะ PatchCore ออกทั้งหมด
(PATCH_POOL_KERNEL, CORESET_RATIO, CORESET_PROJECTION_DIM,
REWEIGHT_NUM_NEIGHBORS, KNN_CHUNK_SIZE) เพราะ DFR ไม่มี memory bank —
แล้วเพิ่ม field เฉพาะ DFR แทน (ดูหัวข้อ "DFR-specific" ด้านล่าง)
"""
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, Optional

import numpy as np
import torch


@dataclass
class Config:
    # ── Data (ต้องตรงกับ repo หลักถ้าจะเทียบผลกัน) ──────────────────
    DATA_ROOT: str = "dataset root path (contains good/ and defect/ subfolders)"
    GOOD_DIRNAME: str = "good"
    DEFECT_DIRNAME: str = "defect"
    SPLIT_RATIOS: Tuple[float, float, float] = (0.70, 0.15, 0.15)
    # ชี้ไปที่ splits/split_assignment.csv ของ repo หลักถ้ามีแล้ว เพื่อ reuse
    # split เดิมเป๊ะๆ (ห้ามลบ/สร้างใหม่ ไม่งั้น train/val/test membership จะ
    # ไม่ตรงกับ baseline อื่น (ConvNeXt+AE, PatchCore, ...) อีกต่อไป
    SPLIT_CACHE_PATH: str = "splits/split_assignment.csv"
    GROUP_ID_REGEX: Optional[str] = None
    VALID_EXT: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")

    SAVE_PATH: str = "save/logs"
    OUTPUT_PATH: str = "save/results"

    # ── Reproducibility ──────────────────────────────────────────────
    SEED: int = 42
    DEVICE: torch.device = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    EXPERIMENT: str = "DFR_Baseline"
    # ใช้โดย io_utils.save_final_results()/output_docs.py เพื่อ label
    # ผลลัพธ์และ README ให้ถูกต้อง (generalization เทียบกับ config.py ของ
    # PatchCore ที่ hardcode "PatchCore" ไว้ตรงๆ ใน io_utils.py — ดู
    # README หัวข้อ "ความต่างจาก PCB-Anomaly-Baselines-PatchCore")
    METHOD_NAME: str = "DFR"

    # ── Image / DataLoader (ต้องตรงกับ repo หลักถ้าจะเทียบผลกัน) ─────
    IMAGE_SIZE: Tuple[int, int] = (224, 224)
    BATCH_SIZE: int = 32
    NUM_WORKERS: int = 2
    PIN_MEMORY: bool = True
    # DFR มี trainable CAE จริง (ต่างจาก PatchCore) — augmentation จึง
    # มีความหมายในทางทฤษฎี แต่ปิดไว้เป็น default เพื่อให้ apples-to-apples
    # กับ baseline อื่นที่ปิดไว้เหมือนกัน เปิดเองได้ถ้าต้องการ ablation
    USE_AUGMENTATION: bool = False
    AUG_COLOR_JITTER: float = 0.20

    # ── Color mode (เหมือน repo หลัก) ────────────────────────────────
    USE_GRAYSCALE: bool = False
    USE_GRAYSCALE_EQUALIZATION: bool = False
    USE_CLAHE: bool = False
    CLAHE_CLIP_LIMIT: float = 2.0
    CLAHE_TILE_GRID_SIZE: tuple = (8, 8)

    # ── Evaluation (ต้องตรงกับ repo หลักถ้าจะเทียบผลกัน) ─────────────
    THRESHOLD_PERCENTILE: float = 95.0
    HEATMAP_SIGMA: float = 4.0

    # ── Backbone ────────────────────────────────────────────────────────
    # "vgg19" คือ backbone ของ paper ต้นฉบับ — ค่า default ของไฟล์นี้
    # ตอนนี้ (ยังไม่ฟันธงว่า thesis จะใช้ backbone ไหนสุดท้าย เก็บ DFR
    # แบบตรงตาม paper ไว้ก่อนเป็น literature reference ล้วนๆ — ดู
    # src/models/dfr.py หัวไฟล์ และ README)
    # สลับไปใช้ 'convnext_tiny' + FEATURE_LAYERS=("features.3","features.5")
    # ได้ทุกเมื่อถ้าตัดสินใจแล้วว่าอยากเทียบ backbone เดียวกับ EXPERIMENT 0/
    # PatchCore/PaDiM แบบ apples-to-apples แทน ('wide_resnet50_2'/
    # 'resnet18'/'resnet50' ก็รองรับเหมือนกัน)
    BACKBONE: str = "vgg19"
    # ต้องเป็น True เสมอตอนรันผลจริง — False มีไว้สำหรับ smoke test/
    # offline dev เท่านั้น (backbone frozen ทั้งตัวเสมอไม่ว่า BACKBONE ไหน)
    PRETRAINED: bool = True
    # Layer ที่ดึง feature ออกมา — default คือ "front 12" ของ VGG19 ทั้ง 16
    # conv layer ตาม paper TABLE I (ReLU output layer 1-12, RF size 3-116
    # พิกเซล) ไม่ใช่ครบ 16 layer เพราะ paper section IV-C.1 เองก็สรุปว่า
    # f_{1:12} ให้ผลใกล้เคียง f_{1:16} มาก (ROC-AUC ต่างกัน <0.01) แต่เบา
    # กว่าอย่างมีนัยสำคัญ (16 layer รวมกัน c_o≈5504 channel ที่ spatial
    # 56x56 หนักมากสำหรับ debug) — ตั้งเป็น
    # `src.models.dfr.VGG19_ALL_16_LAYERS` เองได้ถ้าต้องการ full fidelity
    # ตาม headline result ของ paper (f_{1:16})
    #   ConvNeXt: Stage2="features.3" (192ch,28x28), Stage3="features.5" (384ch,14x14)
    #   ResNet:   ("layer2","layer3") ตาม PatchCore/PaDiM
    FEATURE_LAYERS: Tuple[str, ...] = (
        "features.1",  "features.3",  "features.6",  "features.8",
        "features.11", "features.13", "features.15", "features.17",
        "features.20", "features.22", "features.24", "features.26",
    )

    # Reflection padding (paper section IV-C.2): สลับ Conv2d.padding_mode
    # ของ backbone จาก 'zeros' เป็น 'reflect' — paper เสนอวิธีนี้แก้ปัญหา
    # boundary effect (anomaly หลอกใกล้ขอบภาพจาก zero-padding) และรายงาน
    # ว่าเพิ่ม ROC-AUC/PRO-AUC เฉลี่ยประมาณ 1% เปิดไว้เป็น default (True)
    # ตามที่ paper แนะนำ — ปิดได้ถ้าต้องการ replicate ผลแบบ zero-padding
    # ดั้งเดิม (ค่า default ก่อนหน้าของ torchvision pretrained model)
    DFR_REFLECTION_PADDING: bool = True

    # ── DFR-specific: Regional Feature Generator (paper eq. 1-3) ────────
    # Align (eq. 1): resize ทุก scale ให้เท่ากับขนาดภาพ input ก่อน —
    # "nearest" ตรงกับที่ paper section IV-A.3 ระบุไว้ชัดเจน (ไม่ใช่
    # bilinear)
    DFR_ALIGN_MODE: str = "nearest"
    # Aggregate (eq. 2): mean-filter (avg pool) kernel/stride ต่อทุก scale
    # หลัง align แล้ว — ควบคุมขนาด h_o×w_o ของ regional map สุดท้ายโดยตรง
    # ค่า default 4/4 ตรงกับที่ paper ใช้จริงบน MVTec AD (ภาพ 256x256 →
    # h_o=w_o=64; ที่นี่ IMAGE_SIZE=224 → h_o=w_o=56)
    DFR_AGG_KERNEL: int = 4
    DFR_AGG_STRIDE: int = 4

    # Channel-wise z-score normalization ก่อนเข้า CAE — "ไม่ได้อยู่ใน DFR
    # paper ต้นฉบับ" ปิดไว้เป็น default (False) เพื่อให้ตรง paper เป๊ะ —
    # เปิดได้ (True) ถ้าต้องการให้สอดคล้องกับ convention ของ repo หลัก
    # (mathematical_formulation.md หัวข้อ 3.2) แทน
    NORMALIZE_FEATURES: bool = False

    # จำนวน epoch ที่เทรน CAE — ไม่มี early stopping เพราะ fit() ตาม
    # BaseAnomalyModel interface รับแค่ normal_loader (train) ไม่มี
    # val_loader ให้ monitor ระหว่างเทรน (ต่างจาก repo หลักที่ทำได้เพราะ
    # engine.py เข้าถึง val_loader ตรงๆ) — ดู README หัวข้อ
    # "ความต่างจาก repo หลัก (EXPERIMENT 0)"
    DFR_EPOCHS: int = 100
    DFR_LR: float = 1e-4
    DFR_WEIGHT_DECAY: float = 5e-4
    DFR_LR_STEP: int = 25
    DFR_LR_GAMMA: float = 0.5

    # latent dimension c_d ของ CAE (Appendix B ของ paper) — ถ้าเป็น None
    # จะประมาณอัตโนมัติด้วย PCA บน subset ของ regional feature (90% variance
    # explained ตาม paper section IV-A.3) ถ้าตั้งเป็นตัวเลข จะ fix ค่านั้น
    # ตรงๆ ไม่รัน PCA (เร็วกว่า, ใช้เวลา debug/smoke-test)
    DFR_LATENT_DIM: Optional[int] = None
    DFR_PCA_VARIANCE_RATIO: float = 0.90
    # จำนวน regional feature vector ที่สุ่มมา fit PCA (ทั้งภาพรวมกันมี
    # h_o*w_o*N_images ตัว ซึ่งอาจเยอะเกินจะ fit PCA ได้ในหน่วยความจำเดียว)
    DFR_PCA_SAMPLE_SIZE: int = 50_000

    # การรวม regional anomaly map (h_o x w_o) เป็น image-level score เดียว
    # — ชื่อ/ความหมายเดียวกับ SCORE_METHOD ของ repo หลัก (mean/max/topk)
    # ต่างจาก PatchCore ที่ใช้ "knn"/PaDiM ที่ใช้ "mahalanobis" เพราะ DFR
    # ให้ error map ตรงๆ ไม่ต้อง re-weight แบบ PatchCore
    SCORE_METHOD: str = "topk"
    SCORE_TOPK_RATIO: float = 0.05  # ใช้เมื่อ SCORE_METHOD='topk': สัดส่วน pixel ที่แย่ที่สุดที่เอามาเฉลี่ย

    _DATA_ROOT_PLACEHOLDER = "dataset root path (contains good/ and defect/ subfolders)"

    @property
    def COLOR_MODE(self) -> str:
        if self.USE_GRAYSCALE_EQUALIZATION and self.USE_CLAHE:
            return "GRAYSCALE_EQUALIZATION_CLAHE"
        elif self.USE_GRAYSCALE_EQUALIZATION:
            return "GRAYSCALE_EQUALIZATION"
        elif self.USE_CLAHE:
            return "GRAYSCALE_CLAHE"
        elif self.USE_GRAYSCALE:
            return "GRAYSCALE"
        else:
            return "RGB"

    def __post_init__(self):
        for p in [self.SAVE_PATH, self.OUTPUT_PATH]:
            Path(p).mkdir(parents=True, exist_ok=True)

        ratio_sum = sum(self.SPLIT_RATIOS)
        if not np.isclose(ratio_sum, 1.0, atol=1e-6):
            raise ValueError(
                f"Config.SPLIT_RATIOS must sum to 1.0, got {self.SPLIT_RATIOS} "
                f"(sums to {ratio_sum}).")
        if len(self.SPLIT_RATIOS) != 3:
            raise ValueError(
                f"Config.SPLIT_RATIOS must have exactly 3 values, got "
                f"{len(self.SPLIT_RATIOS)}: {self.SPLIT_RATIOS}")

        if not (0.0 < self.DFR_PCA_VARIANCE_RATIO <= 1.0):
            raise ValueError(
                f"Config.DFR_PCA_VARIANCE_RATIO must be in (0, 1], got "
                f"{self.DFR_PCA_VARIANCE_RATIO}")
        if self.SCORE_METHOD not in ("mean", "max", "topk"):
            raise ValueError(
                f"Config.SCORE_METHOD must be one of 'mean'/'max'/'topk', "
                f"got {self.SCORE_METHOD!r}")
        if self.DFR_ALIGN_MODE not in ("nearest", "bilinear"):
            raise ValueError(
                f"Config.DFR_ALIGN_MODE must be 'nearest' (paper default) or "
                f"'bilinear', got {self.DFR_ALIGN_MODE!r}")
        if self.DFR_AGG_KERNEL < 1 or self.DFR_AGG_STRIDE < 1:
            raise ValueError(
                f"Config.DFR_AGG_KERNEL/DFR_AGG_STRIDE must both be >= 1, got "
                f"kernel={self.DFR_AGG_KERNEL}, stride={self.DFR_AGG_STRIDE}")

        if self.DATA_ROOT == self._DATA_ROOT_PLACEHOLDER:
            raise ValueError(
                "Config.DATA_ROOT is still the default placeholder string. Set "
                "it to a real folder containing "
                f"{self.GOOD_DIRNAME!r} and {self.DEFECT_DIRNAME!r} subfolders, "
                "e.g. DATA_ROOT='/path/to/your/dataset'.\n"
                "แนะนำ: ให้ชี้ไปที่ DATA_ROOT เดียวกับ repo หลัก "
                "(Anomaly-Detection-THESIS) และตั้ง SPLIT_CACHE_PATH ให้ชี้ไปที่ "
                "splits/split_assignment.csv ไฟล์เดียวกัน เพื่อให้ train/val/test "
                "membership ตรงกันเป๊ะระหว่างทุก repo")
        if not Path(self.DATA_ROOT).is_dir():
            raise FileNotFoundError(
                f"Config.DATA_ROOT does not exist or is not a directory: "
                f"{self.DATA_ROOT!r}")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
