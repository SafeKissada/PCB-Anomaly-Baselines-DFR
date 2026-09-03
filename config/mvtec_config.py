"""
Config สำหรับรัน DFR บน MVTec Anomaly Detection dataset (Bergmann et al.
2019) — แยกจาก config/config.py (PCB thesis) เพราะ field ที่ต้องการต่างกัน
โดยพื้นฐาน: MVTec มี train/test ที่กำหนดตายตัวมาแล้ว (ไม่ต้องสุ่ม split
เอง แบบที่ config.py ทำ) และมีหลาย category ให้ loop ผ่าน ไม่ใช่ 1 dataset
เดียว — reuse เฉพาะ field ที่ DFR model (src/models/dfr.py) กับ
AnomalyDataset/build_transforms (src/data/dataset.py) ต้องการจริงๆ เท่านั้น

**ไม่มี field พวก DATA_ROOT/GOOD_DIRNAME/DEFECT_DIRNAME/SPLIT_RATIOS/
SPLIT_CACHE_PATH/THRESHOLD_PERCENTILE** เพราะไม่มีความหมายสำหรับ MVTec —
MVTec ใช้ train/test split ที่มากับ dataset ตรงๆ และ headline metric ของ
paper (ROC-AUC/PRO-AUC ระดับ pixel/region) ไม่ต้องมี classification
threshold เลย (เป็น ranking metric จาก continuous score ล้วนๆ)

Default ทั้งหมดด้านล่างตรงกับ paper section IV-A.3/IV-A.4 ตรงตัว
(ไม่ได้ปรับให้เข้ากับ ConvNeXt/thesis convention แบบ config.py หลัก) —
เป้าหมายของไฟล์นี้คือ "reproduce ตัวเลขในเปเปอร์ให้ใกล้เคียงที่สุด"
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch


# 15 sub-dataset มาตรฐานของ MVTec AD (paper table II/III) — ใช้เป็น default
# ถ้า MVTecConfig.CATEGORIES=None (แปลว่า "รันทุก category ที่มีจริงใน
# MVTEC_ROOT" — ดู src/data/mvtec_dataset.py::discover_categories ซึ่งจะ
# กรองเฉพาะ category ที่มีอยู่จริงเท่านั้น ไม่ error ถ้ามีไม่ครบ 15)
MVTEC_ALL_CATEGORIES = (
    "carpet", "grid", "leather", "tile", "wood",              # textures (5)
    "bottle", "cable", "capsule", "hazelnut", "metal_nut",
    "pill", "screw", "toothbrush", "transistor", "zipper",    # objects (10)
)


@dataclass
class MVTecConfig:
    # ── Data ──────────────────────────────────────────────────────────
    # โฟลเดอร์ที่แตก MVTec AD ไว้แล้ว (ดาวน์โหลด+แตกเองก่อน — ดู README
    # หัวข้อ "MVTec AD reproduction" ว่าทำไม sandbox นี้ดาวน์โหลดให้ไม่ได้)
    # ต้องมีโครงสร้าง: {MVTEC_ROOT}/{category}/train/good/*, .../test/*/*,
    # .../ground_truth/{defect_type}/*_mask.png ตามที่ MVTec AD แจกจริง
    MVTEC_ROOT: str = "path to extracted MVTec AD root (contains carpet/, grid/, ... subfolders)"
    # None = รันทุก category ที่เจอจริงใน MVTEC_ROOT (เทียบกับ
    # MVTEC_ALL_CATEGORIES เป็น allowlist กันโฟลเดอร์ขยะปนเข้ามา) ระบุ
    # list เช่น ["carpet","grid"] ถ้าต้องการรันบางหมวดเพื่อ debug ให้เร็ว
    CATEGORIES: Optional[Tuple[str, ...]] = None

    SAVE_PATH: str = "save/mvtec/logs"
    OUTPUT_PATH: str = "save/mvtec/table"

    # ── Reproducibility ───────────────────────────────────────────────
    SEED: int = 42
    DEVICE: torch.device = None  # ตั้งใน __post_init__ (ไม่ใช้ default_factory เพราะต้อง log ด้วย)
    EXPERIMENT: str = "DFR_MVTec_reproduction"
    METHOD_NAME: str = "DFR"

    # ── Image / DataLoader — ตรงตาม paper section IV-A.4 เป๊ะ ──────────
    # "the images are resized to the size of 256 × 256 pixels" (paper IV-A.4)
    IMAGE_SIZE: Tuple[int, int] = (256, 256)
    # "a batch size of 4" (paper IV-A.4)
    BATCH_SIZE: int = 4
    NUM_WORKERS: int = 2
    PIN_MEMORY: bool = True
    # color-mode fields ที่ build_transforms() (reuse จาก src/data/dataset.py)
    # ต้องการ — MVTec เป็นภาพสี/เทาปนกันตามหมวด แต่ dataset โหลดเป็น RGB
    # เสมอผ่าน .convert("RGB") อยู่แล้ว (ดู AnomalyDataset._load_one) จึงไม่
    # จำเป็นต้องแตะ color-mode ใดๆ เพิ่ม คงไว้เป็น RGB ตรงไปตรงมา
    USE_GRAYSCALE: bool = False
    USE_GRAYSCALE_EQUALIZATION: bool = False
    USE_CLAHE: bool = False
    CLAHE_CLIP_LIMIT: float = 2.0
    CLAHE_TILE_GRID_SIZE: tuple = (8, 8)
    USE_AUGMENTATION: bool = False   # paper ไม่ augment
    AUG_COLOR_JITTER: float = 0.20   # unused ถ้า USE_AUGMENTATION=False แต่ build_transforms() ต้องการ field นี้อยู่ดี

    # ── Backbone — VGG19 ตาม paper (ไม่ใช่ ConvNeXt แบบ RUN.py หลัก) ────
    BACKBONE: str = "vgg19"
    PRETRAINED: bool = True  # ต้อง True เสมอตอน reproduce จริง (internet ต้องโหลด ImageNet weight ได้)
    # front-12 (paper section IV-C.1: ผลใกล้เคียง f{1:16} มาก แต่เบากว่า
    # มาก) — ตั้งเป็น src.models.dfr.VGG19_ALL_16_LAYERS เองได้ถ้าต้องการ
    # headline number เต็มรูปแบบตาม Table II/III คอลัมน์ "Ours f{1:16}"
    FEATURE_LAYERS: Tuple[str, ...] = (
        "features.1",  "features.3",  "features.6",  "features.8",
        "features.11", "features.13", "features.15", "features.17",
        "features.20", "features.22", "features.24", "features.26",
    )
    # ── DFR-specific — ตรงตาม paper ทุก field (ดู src/models/dfr.py) ────
    DFR_ALIGN_MODE: str = "nearest"        # paper IV-A.3: nearest-neighbor
    DFR_AGG_KERNEL: int = 4                # paper IV-A.3: "mean filter ... 4×4 ... stride of 4"
    DFR_AGG_STRIDE: int = 4
    DFR_REFLECTION_PADDING: bool = True    # paper IV-C.2's own fix, ~1% avg improvement
    NORMALIZE_FEATURES: bool = False       # paper ไม่ normalize feature ก่อนเข้า CAE
    # "a learning rate of 1×10−4 ... for 700 epochs" (paper IV-A.4)
    DFR_EPOCHS: int = 700
    DFR_LR: float = 1e-4
    DFR_WEIGHT_DECAY: float = 5e-4         # ไม่ได้ระบุใน paper — ค่า default เดียวกับ config.py หลัก
    DFR_LR_STEP: int = 175                 # ไม่ได้ระบุใน paper (paper ไม่ได้บอกว่ามี LR schedule เลยด้วยซ้ำ) —
    DFR_LR_GAMMA: float = 0.5              # ตั้งไว้ให้ decay 2 ครั้งตลอด 700 epoch แทนค่าคงที่ตลอด ปรับได้/ปิดได้ (step=DFR_EPOCHS+1)
    # latent dim c_d — paper: "estimate ... with PCA such that 90% variance
    # is just explained" (IV-A.3) — None = ประมาณอัตโนมัติ
    DFR_LATENT_DIM: Optional[int] = None
    DFR_PCA_VARIANCE_RATIO: float = 0.90
    DFR_PCA_SAMPLE_SIZE: int = 50_000
    # จำกัด peak GPU memory ของขั้น align — batch size ของ paper (4) เล็ก
    # อยู่แล้ว ไม่ค่อยจำเป็นต้อง chunk ต่อ แต่เผื่อไว้เผื่อมีคนเพิ่ม
    # BATCH_SIZE เอง
    DFR_FEATURE_CHUNK_SIZE: Optional[int] = 4

    # SCORE_METHOD/SCORE_TOPK_RATIO ยังต้องมี (DFR.score() ใช้เสมอ) แต่ไม่มี
    # ผลต่อ headline metric ของ MVTec (ROC-AUC/PRO-AUC เป็น pixel/region
    # metric ล้วนๆ ไม่ใช้ image-level aggregate score เลย — ดู
    # scripts/run_dfr_mvtec.py) เก็บไว้เพราะ DFR.score() คำนวณคู่กันเสมอ
    SCORE_METHOD: str = "max"
    SCORE_TOPK_RATIO: float = 0.05
    # HEATMAP_SIGMA — paper section III-D ไม่ได้ใช้ Gaussian smoothing บน
    # anomaly map เลย (แค่ bilinear upsample ตรงๆ) ตั้งเป็น 0 (ไม่ smooth)
    # ให้ตรง paper เป๊ะ — ต่างจาก config.py หลักที่ default=4.0 (เพิ่มเพื่อ
    # ให้ heatmap ดูสวยตอน visualize เฉยๆ ไม่ใช่ตาม paper)
    HEATMAP_SIGMA: float = 0.0

    # ── PRO-AUC evaluation protocol (paper IV-A.5) ─────────────────────
    # "we report the normalized PRO-AUC up to an average per-pixel false
    # positive rate (FPR) of 30%" — ตรงตัว
    PRO_MAX_FPR: float = 0.30
    PRO_NUM_THRESHOLDS: int = 200  # ความละเอียดของ threshold sweep — ดู src/mvtec_evaluate.py

    def __post_init__(self):
        if self.DEVICE is None:
            self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for p in [self.SAVE_PATH, self.OUTPUT_PATH]:
            Path(p).mkdir(parents=True, exist_ok=True)

        if not (0.0 < self.DFR_PCA_VARIANCE_RATIO <= 1.0):
            raise ValueError(
                f"MVTecConfig.DFR_PCA_VARIANCE_RATIO must be in (0, 1], got "
                f"{self.DFR_PCA_VARIANCE_RATIO}")
        if self.DFR_ALIGN_MODE not in ("nearest", "bilinear"):
            raise ValueError(
                f"MVTecConfig.DFR_ALIGN_MODE must be 'nearest' or 'bilinear', "
                f"got {self.DFR_ALIGN_MODE!r}")
        if not (0.0 < self.PRO_MAX_FPR <= 1.0):
            raise ValueError(
                f"MVTecConfig.PRO_MAX_FPR must be in (0, 1], got {self.PRO_MAX_FPR}")

        placeholder = "path to extracted MVTec AD root (contains carpet/, grid/, ... subfolders)"
        if self.MVTEC_ROOT == placeholder:
            raise ValueError(
                "MVTecConfig.MVTEC_ROOT is still the default placeholder. Set it "
                "to a real local folder containing the extracted MVTec AD dataset "
                "(e.g. MVTEC_ROOT='/path/to/mvtec_anomaly_detection').\n"
                "สำคัญ: ไฟล์นี้ไม่ได้ดาวน์โหลด MVTec AD ให้อัตโนมัติ — ต้องโหลด+"
                "แตกไฟล์เองก่อน (ดู README หัวข้อ 'MVTec AD reproduction' "
                "สำหรับ official/mirror link และเหตุผลว่าทำไม sandbox นี้ทำให้ "
                "ไม่ได้)")
        if not Path(self.MVTEC_ROOT).is_dir():
            raise FileNotFoundError(
                f"MVTecConfig.MVTEC_ROOT does not exist or is not a directory: "
                f"{self.MVTEC_ROOT!r}")
