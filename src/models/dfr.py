"""
DFR (Yang, Shi & Qi, "DFR: Deep Feature Reconstruction for Unsupervised
Anomaly Segmentation", arXiv:2012.07122) — https://arxiv.org/abs/2012.07122

**เวอร์ชันนี้ implement ตาม paper ต้นฉบับให้ใกล้เคียงที่สุด** (backbone,
align-then-aggregate, ไม่ normalize) แทนที่จะปรับให้เข้ากับ convention ของ
thesis (ConvNeXt, z-score) เหมือนฉบับก่อนหน้า — เพราะยังไม่ได้ฟันธงว่า
thesis จะใช้ backbone ไหนหรือจะทำ normalization แบบไหน การมี DFR แบบ
"ตรงตาม paper" ไว้ก่อนทำให้เห็นผลของ literature method ล้วนๆ โดยไม่ปนกับ
choice ของ thesis เอง — ถ้าภายหลังตัดสินใจแล้วว่า thesis จะใช้ backbone/
normalization แบบไหน ค่อยปรับ cfg.BACKBONE/cfg.FEATURE_LAYERS/
cfg.NORMALIZE_FEATURES ให้ตรงกันเพื่อเทียบแบบ apples-to-apples ก็ยังทำได้
(ทุก field เป็น config, ไม่ hardcode)

สรุป pipeline (ตาม paper section III):
  1. Hierarchical Image Feature Extraction: ดึง feature map จาก 16
     convolutional layer (ReLU output) ของ VGG19 pretrained บน ImageNet
     แบบ frozen ทั้งตัว (paper section IV-A.3) — ตาราง RF size ของแต่ละ
     layer อยู่ใน TABLE I ของ paper (3 ถึง 252 พิกเซล)
  2. Multi-scale Regional Feature Generation (paper eq. 1-3):
       a) Align: resize ทุก scale ให้เท่ากับ "ขนาดภาพ input" (h×w) ด้วย
          nearest-neighbor interpolation (ไม่ใช่ bilinear — paper ระบุ
          ชัดเจนว่าใช้ nearest-neighbor)
       b) Aggregate: mean filter (avg pooling) kernel=cfg.DFR_AGG_KERNEL
          stride=cfg.DFR_AGG_STRIDE ต่อทุก scale — ทำหน้าที่ 2 อย่าง:
          (i) smooth ให้ feature ทนต่อ noise มากขึ้น (ii) ควบคุมขนาด
          h_o×w_o ของ regional map สุดท้าย (ค่า default kernel=stride=4
          ตรงกับที่ paper ใช้จริงบน MVTec AD: ภาพ 256×256 → h_o=w_o=64)
       c) Concatenate: รวมทุก scale ตามแกน channel ได้ f(x) ขนาด
          [h_o, w_o, c_o] เดียว, c_o = ผลรวม channel ของทุก scale
  3. Deep Feature Reconstruction: CAE ที่มีแต่ 1x1 conv + ReLU (Appendix B
     ของ paper, 6 layer) เทรนบน regional feature ของภาพ normal เท่านั้น
     ด้วย reconstruction loss L2 แบบ pair-wise (paper eq. 4)
  4. Anomaly Scoring: error map A(x) = ||f(x) - f_hat(x)||_2 ต่อตำแหน่ง
     (paper eq. 5) → upsample เป็น pixel-wise anomaly map → รวมเป็น
     image-level score เดียวด้วย cfg.SCORE_METHOD (mean/max/topk — paper
     ไม่ได้ระบุการรวมเป็น image-level score ไว้ชัด เพราะ paper เน้น
     pixel/region-level segmentation metric (ROC-AUC/PRO-AUC) เป็นหลัก
     ไม่ใช่ image-level classification เหมือน thesis นี้)

หมายเหตุสำคัญสำหรับ dataset นี้ (ต่างจาก MVTec ที่ paper ต้นฉบับ benchmark
และต่างจาก repo หลัก EXPERIMENT 0):
  - CAE เทรนจาก "Good" (=false call) เท่านั้น ซึ่งตาม context เป็น
    biased/narrow sample ของ normal ไม่ใช่ภาพงานดีทั่วไปที่ไม่ถูก AOI flag
    เลย — ตีความผลลัพธ์โดยเผื่อ distribution-shift นี้ไว้เสมอ
  - fit() ตาม BaseAnomalyModel interface รับแค่ normal_loader ไม่มี
    val_loader ให้ monitor ระหว่างเทรน จึงเทรนแบบ fixed-epoch schedule
    (StepLR) ไม่มี early stopping — ต่างจาก EXPERIMENT 0 ที่ engine.py
    เข้าถึง val_loader ได้ตรงๆ (ดู README)
  - latent dimension c_d ประมาณอัตโนมัติด้วย PCA (90% variance, ตาม paper
    section IV-A.3) เว้นแต่ตั้ง cfg.DFR_LATENT_DIM ไว้ตรงๆ
  - image-level classification (accuracy/AUROC ของทั้งภาพ) ไม่ใช่สิ่งที่
    paper วัดหลัก (paper วัด pixel/region segmentation) — SCORE_METHOD
    (mean/max/topk) เป็นส่วนที่ "เพิ่มเข้ามา" เพื่อให้ได้ image-level score
    เดียวสำหรับเทียบกับ EXPERIMENT 0/PatchCore ในตาราง comparison เดียวกัน

ยังต่างจาก paper ต้นฉบับอยู่ 1 จุด (ระบุไว้ตรงๆ):
  - Backbone หลัก MVTec AD ในกระดาษของ paper คือ VGG19 (fixed) — repo นี้
    ให้ VGG19 เป็นค่า default (ตรงกับ paper) แต่ยังคง config
    BACKBONE/FEATURE_LAYERS ให้สลับไปใช้ ConvNeXt/ResNet ได้ (สำหรับตอนที่
    ตัดสินใจแล้วว่าอยากเทียบ backbone เดียวกับ thesis) — ไม่ใช่ hardcode
    ตายตัว

Reflection padding (paper section IV-C.2, "Boundary Effects"): เปิดโดย
default ผ่าน cfg.DFR_REFLECTION_PADDING=True — สลับ Conv2d.padding_mode
ของ backbone ทั้งตัวจาก 'zeros' เป็น 'reflect' แก้ปัญหา anomaly หลอกใกล้
ขอบภาพจาก zero-padding (paper รายงานว่าเพิ่ม ROC-AUC/PRO-AUC เฉลี่ย ~1%)
พิสูจน์เพิ่มเติมระหว่างพัฒนา repo นี้ว่าจำเป็นจริง ไม่ใช่แค่ทางเลือก
เสริม — ดู README หัวข้อ "Reflection padding" และ "Heatmap sanity check
finding" สำหรับรายละเอียดการตรวจสอบเต็มรูปแบบ (รวมถึงข้อจำกัดที่ยัง
เหลืออยู่เมื่อใช้ backbone แบบ random-init/PRETRAINED=False)
"""
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

logger = logging.getLogger("DFR")

# ── VGG19 — 16 conv(+ReLU) layer names ตาม paper TABLE I (RF size 3→252) ──
# ตรวจสอบจาก torchvision.models.vgg19().features จริง (ไม่ได้เดา) — index
# ที่นี่คือ ReLU module ที่ต่อจาก conv แต่ละตัว (paper: "we get CNN feature
# maps from the ReLU outputs of the convolutional layers")
VGG19_ALL_16_LAYERS: Tuple[str, ...] = (
    "features.1",  "features.3",  "features.6",  "features.8",
    "features.11", "features.13", "features.15", "features.17",
    "features.20", "features.22", "features.24", "features.26",
    "features.29", "features.31", "features.33", "features.35",
)
# paper section IV-C.1: f_{1:12} ("front 12 scales") ให้ผลใกล้เคียง f_{1:16}
# มาก แต่เบากว่า — ทางเลือกถ้า VGG19_ALL_16_LAYERS หนักเกินไปสำหรับ
# เครื่องที่มี — ตั้ง cfg.FEATURE_LAYERS=VGG19_FRONT_12_LAYERS แทนได้
VGG19_FRONT_12_LAYERS: Tuple[str, ...] = VGG19_ALL_16_LAYERS[:12]

# ── Backbone factory ──────────────────────────────────────────────────
# ConvNeXt/ResNet ยังอยู่ในนี้เผื่อภายหลังตัดสินใจว่า thesis จะเทียบ
# backbone เดียวกันข้าม method (ดู docstring ด้านบน) — ConvNeXt stage
# mapping เหมือนกับ src/models/patchcore.py ของ
# PCB-Anomaly-Baselines-PatchCore ทุกประการ
_BACKBONE_FACTORY = {
    "wide_resnet50_2": (torchvision.models.wide_resnet50_2,
                         torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V2),
    "resnet18": (torchvision.models.resnet18,
                 torchvision.models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet50": (torchvision.models.resnet50,
                 torchvision.models.ResNet50_Weights.IMAGENET1K_V2),
    # ConvNeXt stage boundary -> module name mapping (สำหรับ cfg.FEATURE_LAYERS):
    #   Stage2 (192ch) -> "features.3"
    #   Stage3 (384ch) -> "features.5"
    #   Stage4 (768ch) -> "features.7"
    "convnext_tiny": (torchvision.models.convnext_tiny,
                       torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT),
    # VGG19 — backbone ของ paper ต้นฉบับ, ค่า default ของ repo นี้ (ดู
    # VGG19_ALL_16_LAYERS ด้านบนสำหรับชื่อ layer ที่ใช้กับ FEATURE_LAYERS)
    "vgg19": (torchvision.models.vgg19,
              torchvision.models.VGG19_Weights.IMAGENET1K_V1),
}


class _FeatureExtractor(torch.nn.Module):
    """Frozen backbone + forward hooks บน cfg.FEATURE_LAYERS — ไม่มี
    trainable parameter เลย (ต่างจาก CAE ด้านล่างที่เทรนได้) เหมือน
    _FeatureExtractor ของ patchcore.py ทุกประการ ยกเว้น 1 จุดเพิ่มเติม:
    reflection-padding switch (ดู use_reflection_padding ด้านล่าง)

    **Boundary effects / reflection padding (paper section IV-C.2)**: paper
    ต้นฉบับรายงานว่า zero-padding ของ backbone ทำให้เกิด anomaly หลอกใกล้
    ขอบภาพ (โดยเฉพาะภาพที่ foreground เต็มภาพ เช่น texture category) และ
    เสนอ reflection padding เป็นวิธีแก้ (รายงานว่าเพิ่ม ROC-AUC/PRO-AUC
    เฉลี่ยประมาณ 1%) — พิสูจน์เพิ่มเติมระหว่างพัฒนา repo นี้ด้วย diagnostic
    script (ดู README หัวข้อ "Reflection padding") ว่าด้วย random-init
    backbone (สำหรับ smoke test) ผลกระทบนี้รุนแรงกว่ามาก: correlation
    ระหว่าง feature-magnitude map ของภาพสุ่ม 2 ภาพที่ไม่เกี่ยวข้องกันเลย
    สูงถึง 0.95 (แปลว่า error map เกือบทั้งหมดมาจากตำแหน่ง ไม่ใช่เนื้อหา
    ภาพ) — สลับ Conv2d.padding_mode เป็น 'reflect' ลดค่านี้เหลือ 0.04
    """

    def __init__(self, backbone_name: str, layers: Tuple[str, ...], device,
                 pretrained: bool = True, use_reflection_padding: bool = True):
        super().__init__()
        if backbone_name not in _BACKBONE_FACTORY:
            raise ValueError(
                f"Unknown BACKBONE {backbone_name!r}. Supported: "
                f"{list(_BACKBONE_FACTORY)}")
        ctor, weights = _BACKBONE_FACTORY[backbone_name]
        if not pretrained:
            logger.warning(
                "pretrained=False: ใช้ random-init weights — สำหรับ smoke "
                "test/offline dev เท่านั้น ห้ามใช้รันผลจริงเด็ดขาด")
        self.backbone = ctor(weights=weights if pretrained else None)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        if use_reflection_padding:
            n_switched = 0
            for m in self.backbone.modules():
                if (isinstance(m, nn.Conv2d) and m.padding_mode == "zeros"
                        and (m.padding[0] > 0 or m.padding[1] > 0)):
                    m.padding_mode = "reflect"
                    n_switched += 1
            logger.info(f"Reflection padding (paper §IV-C.2): switched "
                        f"{n_switched} Conv2d layers from zero->reflect padding")

        self.backbone.to(device)

        self.layers = layers
        self._features = {}
        self._hooks = []
        for name in layers:
            module = dict(self.backbone.named_modules())[name]
            self._hooks.append(
                module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name):
        def hook(_module, _input, output):
            self._features[name] = output
        return hook

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> "list[torch.Tensor]":
        self._features = {}
        self.backbone(x)
        return [self._features[name] for name in self.layers]

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()


from dataclasses import dataclass


@dataclass(frozen=True)
class RegionalFeatureParams:
    """รวม hyperparameter ของ regional feature generator (paper eq. 1-3)
    ไว้ก้อนเดียว กันต้องส่ง 4 argument แยกกันทุกที่ที่เรียก
    _regional_feature_map()
    """
    align_mode: str      # "nearest" (paper) หรือ "bilinear"
    image_size: Tuple[int, int]  # (h, w) ที่ align ทุก scale เข้าไปหา (paper eq. 1)
    agg_kernel: int       # kernel ของ mean-filter aggregation (paper eq. 2)
    agg_stride: int       # stride ของ mean-filter aggregation — คุมขนาด h_o,w_o


@torch.no_grad()
def _regional_feature_map(extractor: _FeatureExtractor, images: torch.Tensor,
                           params: RegionalFeatureParams) -> torch.Tensor:
    """สร้าง multi-scale regional feature map f(x) ตาม paper eq. 1-3 (align
    → aggregate → concatenate):

      1) Align (eq. 1): resize ทุก scale ให้เท่ากับ "ขนาดภาพ input"
         (params.image_size) ด้วย nearest-neighbor interpolation (ตาม
         paper section IV-A.3 ที่ระบุชัดว่าใช้ nearest-neighbor ไม่ใช่
         bilinear) — ได้ φ̂_l(x) ขนาด h×w×c_l ทุก scale เท่ากันหมด

      2) Aggregate (eq. 2): mean filter (avg pooling) kernel=agg_kernel
         stride=agg_stride ต่อทุก scale ที่ align แล้ว — smooth feature
         และย่อขนาดจาก h×w ลงเหลือ h_o×w_o พร้อมกัน (ค่า default
         kernel=stride=4 ตรงกับที่ paper ใช้จริงบน MVTec AD)

      3) Concatenate (eq. 3): รวมทุก scale ตามแกน channel → f(x) ขนาด
         [B, c_o, h_o, w_o] เดียว, c_o = ผลรวม channel ของทุก scale

    คืน [B, C_total, h_o, w_o]
    """
    feats = extractor(images)  # list of [B, C_l, H_l, W_l], คนละ resolution กัน
    h, w = params.image_size
    align_kwargs = {} if params.align_mode == "nearest" else {"align_corners": False}

    aggregated = []
    for f in feats:
        # 1) align: resize ให้เท่ากับขนาดภาพ input เสมอ (ไม่ใช่ resize
        # ไปหา scale ที่ละเอียดที่สุดแบบเวอร์ชันก่อนหน้า — นี่คือจุดที่
        # แก้ให้ตรงกับ paper eq. 1 จริงๆ)
        aligned = F.interpolate(f, size=(h, w), mode=params.align_mode,
                                 **align_kwargs)
        # 2) aggregate: mean filter ควบคุมขนาด h_o x w_o โดยตรง — ไม่มี
        # padding (paper ไม่ได้ระบุ padding mode ไว้ ใช้ default=0 ของ
        # avg_pool2d ซึ่งตัดขอบที่ไม่ครบ kernel ทิ้งไปเฉยๆ)
        agg = F.avg_pool2d(aligned, kernel_size=params.agg_kernel,
                            stride=params.agg_stride)
        aggregated.append(agg)

    return torch.cat(aggregated, dim=1)  # [B, C_total, h_o, w_o]


class _ChannelZScore:
    """Channel-wise z-score normalization ของ regional feature — fit จาก
    ภาพ normal (train) เท่านั้น (mathematical_formulation.md หัวข้อ 3.2 ของ
    repo หลัก) ปิดได้ผ่าน cfg.NORMALIZE_FEATURES=False (ดู README)

    ใช้สูตร streaming E[X^2] - (E[X])^2 เพื่อไม่ต้องเก็บ feature ทั้งหมด
    ไว้ในหน่วยความจำพร้อมกัน — เดียวกับที่ repo หลักใช้
    """

    def __init__(self):
        self.mean: Optional[torch.Tensor] = None  # [C]
        self.std: Optional[torch.Tensor] = None   # [C]

    @torch.no_grad()
    def fit(self, extractor, loader, device, params: RegionalFeatureParams) -> None:
        n = 0
        s1 = None  # sum
        s2 = None  # sum of squares
        for batch in loader:
            images = batch[0].to(device)
            feat = _regional_feature_map(extractor, images, params)  # [B,C,H,W]
            b, c, h, w = feat.shape
            flat = feat.permute(1, 0, 2, 3).reshape(c, -1)  # [C, B*H*W]
            if s1 is None:
                s1 = flat.sum(dim=1)
                s2 = (flat ** 2).sum(dim=1)
            else:
                s1 += flat.sum(dim=1)
                s2 += (flat ** 2).sum(dim=1)
            n += flat.shape[1]
        mean = s1 / n
        var = torch.clamp(s2 / n - mean ** 2, min=1e-8)
        self.mean = mean
        self.std = torch.sqrt(var)

    def apply(self, feat: torch.Tensor) -> torch.Tensor:
        if self.mean is None:
            return feat
        c = feat.shape[1]
        mean = self.mean.view(1, c, 1, 1)
        std = self.std.view(1, c, 1, 1)
        return (feat - mean) / std


class _DFRCae(nn.Module):
    """Convolutional autoencoder — Appendix B ของ paper: 6 layer, มีแต่
    1x1 conv + ReLU (ไม่มี pooling/stride ใดๆ เพราะทำงานบน "channel"
    ของแต่ละตำแหน่ง spatial อิสระต่อกัน ไม่ใช่ spatial conv จริง)

    [layer 1] Conv(1,1,(c_o+c_d)//2), ReLU
    [layer 2] Conv(1,1,2*c_d), ReLU
    [layer 3] Conv(1,1,c_d)            <- bottleneck, ไม่มี ReLU (paper)
    [layer 4] Conv(1,1,2*c_d), ReLU
    [layer 5] Conv(1,1,(c_o+c_d)//2), ReLU
    [layer 6] Conv(1,1,c_o)            <- reconstruction, ไม่มี ReLU
    """

    def __init__(self, c_in: int, c_latent: int):
        super().__init__()
        c_mid = max(1, (c_in + c_latent) // 2)
        c_2lat = max(1, 2 * c_latent)
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_mid, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(c_mid, c_2lat, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(c_2lat, c_latent, kernel_size=1),
            nn.Conv2d(c_latent, c_2lat, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(c_2lat, c_mid, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(c_mid, c_in, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@torch.no_grad()
def _estimate_latent_dim(extractor, loader, device, params: RegionalFeatureParams,
                          variance_ratio: float, sample_size: int,
                          znorm: Optional[_ChannelZScore]) -> int:
    """ประมาณ latent dimension c_d ด้วย PCA บน subset ของ regional feature
    vector (paper section IV-A.3: "estimate the latent code dimension with
    PCA such that 90% variance is just explained") — สุ่มเก็บ vector จาก
    หลายภาพ/หลายตำแหน่งจนครบ sample_size แล้ว fit PCA ครั้งเดียว
    """
    from sklearn.decomposition import PCA

    collected = []
    n_collected = 0
    for batch in loader:
        if n_collected >= sample_size:
            break
        images = batch[0].to(device)
        feat = _regional_feature_map(extractor, images, params)  # [B,C,H,W]
        if znorm is not None:
            feat = znorm.apply(feat)
        b, c, h, w = feat.shape
        flat = feat.permute(0, 2, 3, 1).reshape(-1, c).cpu().numpy()  # [B*H*W, C]
        take = min(len(flat), sample_size - n_collected)
        idx = np.random.choice(len(flat), size=take, replace=False)
        collected.append(flat[idx])
        n_collected += take

    X = np.concatenate(collected, axis=0)
    pca = PCA(n_components=min(X.shape), svd_solver="full")
    pca.fit(X)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    c_d = int(np.searchsorted(cum_var, variance_ratio) + 1)
    c_d = max(1, min(c_d, X.shape[1]))
    logger.info(f"PCA latent-dim estimate: {c_d} components explain "
                f"{cum_var[c_d - 1]:.4f} variance (target={variance_ratio})")
    return c_d


def _aggregate_score(error_map: torch.Tensor, method: str, topk_ratio: float) -> float:
    """รวม regional error map (h_o x w_o) เป็น image-level score เดียว —
    เดียวกับความหมายของ SCORE_METHOD ใน repo หลัก (mean/max/topk)
    """
    flat = error_map.flatten()
    if method == "mean":
        return float(flat.mean())
    if method == "max":
        return float(flat.max())
    if method == "topk":
        k = max(1, int(len(flat) * topk_ratio))
        return float(torch.topk(flat, k).values.mean())
    raise ValueError(f"Unknown SCORE_METHOD {method!r}")


class DFR:
    """ใช้ตาม interface ของ BaseAnomalyModel (fit / score) แต่ไม่ inherit
    ตรงๆ เพื่อเลี่ยง import cycle กับ base.py (เดียวกับ pattern ของ
    patchcore.py) — ดู scripts/run_dfr.py ว่าประกอบเข้ากับ ScoreResult
    ยังไง
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.DEVICE
        self.extractor = _FeatureExtractor(
            cfg.BACKBONE, cfg.FEATURE_LAYERS, self.device,
            pretrained=getattr(cfg, "PRETRAINED", True),
            use_reflection_padding=cfg.DFR_REFLECTION_PADDING)
        # paper eq. 1-2: align ทุก scale ให้เท่ากับขนาดภาพ input แล้ว
        # aggregate ด้วย mean filter — ประกอบพารามิเตอร์ไว้ครั้งเดียวตรงนี้
        # ใช้ซ้ำทุกที่ที่เรียก _regional_feature_map()
        self.rf_params = RegionalFeatureParams(
            align_mode=cfg.DFR_ALIGN_MODE,
            image_size=tuple(cfg.IMAGE_SIZE),
            agg_kernel=cfg.DFR_AGG_KERNEL,
            agg_stride=cfg.DFR_AGG_STRIDE,
        )
        self.znorm = _ChannelZScore() if cfg.NORMALIZE_FEATURES else None
        self.cae: Optional[_DFRCae] = None
        self.embed_spatial_shape: Optional[Tuple[int, int]] = None
        self.c_in: Optional[int] = None
        self.latent_dim: Optional[int] = None
        self.history = {"train_loss": []}

    def fit(self, normal_loader) -> None:
        """เทรน CAE จากภาพ normal เท่านั้น — ห้ามแตะ defect label ใดๆ

        1) (ถ้าเปิด NORMALIZE_FEATURES) fit channel-wise z-score stats
        2) ประมาณ/กำหนด latent dim c_d
        3) เทรน CAE ด้วย reconstruction loss L2 (paper eq. 4) เป็นเวลา
           cfg.DFR_EPOCHS epoch แบบ fixed schedule (StepLR) — ไม่มี early
           stopping เพราะไม่มี val_loader ให้ monitor (ดู README)
        """
        logger.info("DFR.fit(): extracting regional features จากภาพ normal ทั้งหมด...")

        # ── 1) channel z-score (optional, ปิดโดย default — paper ไม่ทำ) ──
        if self.znorm is not None:
            logger.info("Fitting channel-wise z-score stats จาก train-normal...")
            self.znorm.fit(self.extractor, normal_loader, self.device,
                           self.rf_params)

        # ── infer c_in + spatial shape จาก 1 batch ─────────────────────
        first_batch = next(iter(normal_loader))
        with torch.no_grad():
            probe = _regional_feature_map(
                self.extractor, first_batch[0].to(self.device), self.rf_params)
        self.c_in = probe.shape[1]
        self.embed_spatial_shape = tuple(probe.shape[-2:])

        # ── 2) latent dim ────────────────────────────────────────────
        if self.cfg.DFR_LATENT_DIM is not None:
            self.latent_dim = int(self.cfg.DFR_LATENT_DIM)
            logger.info(f"Using fixed DFR_LATENT_DIM={self.latent_dim} "
                        f"(PCA estimation skipped)")
        else:
            self.latent_dim = _estimate_latent_dim(
                self.extractor, normal_loader, self.device, self.rf_params,
                self.cfg.DFR_PCA_VARIANCE_RATIO, self.cfg.DFR_PCA_SAMPLE_SIZE,
                self.znorm)

        logger.info(f"CAE: c_in={self.c_in}  c_latent={self.latent_dim}  "
                    f"spatial={self.embed_spatial_shape}")
        self.cae = _DFRCae(self.c_in, self.latent_dim).to(self.device)

        # ── 3) training loop (fixed epoch, no early stopping) ──────────
        optim = torch.optim.Adam(self.cae.parameters(), lr=self.cfg.DFR_LR,
                                  weight_decay=self.cfg.DFR_WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.StepLR(
            optim, step_size=self.cfg.DFR_LR_STEP, gamma=self.cfg.DFR_LR_GAMMA)

        self.cae.train()
        for epoch in range(self.cfg.DFR_EPOCHS):
            epoch_loss, n_batches = 0.0, 0
            for batch in normal_loader:
                images = batch[0].to(self.device)
                with torch.no_grad():
                    feat = _regional_feature_map(
                        self.extractor, images, self.rf_params)
                    if self.znorm is not None:
                        feat = self.znorm.apply(feat)

                recon = self.cae(feat)
                # paper eq. 4: sum of pair-wise L2 distance over spatial
                # positions, averaged over the batch — reduction='mean'
                # ทำหน้าที่เดียวกันในทางปฏิบัติ (ค่า scale ต่างกันแค่
                # ค่าคงที่ ไม่กระทบทิศทาง gradient)
                loss = F.mse_loss(recon, feat, reduction="mean")

                optim.zero_grad()
                loss.backward()
                optim.step()

                epoch_loss += float(loss.item())
                n_batches += 1
            sched.step()

            avg_loss = epoch_loss / max(1, n_batches)
            self.history["train_loss"].append(avg_loss)
            if (epoch + 1) % max(1, self.cfg.DFR_EPOCHS // 10) == 0 or epoch == 0:
                logger.info(f"[epoch {epoch + 1}/{self.cfg.DFR_EPOCHS}] "
                            f"train_loss={avg_loss:.6f}  lr={sched.get_last_lr()[0]:.2e}")

        self.cae.eval()
        logger.info("DFR.fit() เสร็จสิ้น — CAE พร้อมใช้")

    @torch.no_grad()
    def score(self, loader):
        from src.models.base import ScoreResult  # lazy import กัน circular import

        if self.cae is None:
            raise RuntimeError("DFR.score() ถูกเรียกก่อน fit() — ต้องเทรน "
                                "CAE จากภาพ normal ก่อนเสมอ")

        image_scores, y_true, labels, paths = [], [], [], []
        pixel_maps, orig_imgs, preproc_imgs = [], [], []
        H, W = self.embed_spatial_shape

        for batch in loader:
            images, orig, preproc, batch_paths, batch_labels, _size = batch
            images = images.to(self.device)
            B = images.shape[0]

            feat = _regional_feature_map(self.extractor, images, self.rf_params)
            assert tuple(feat.shape[-2:]) == (H, W), (
                f"Spatial shape เปลี่ยนระหว่าง fit() ({H},{W}) กับ score() "
                f"({tuple(feat.shape[-2:])}) — เช็คว่า IMAGE_SIZE ตรงกันทั้งสองรอบ")
            if self.znorm is not None:
                feat = self.znorm.apply(feat)

            recon = self.cae(feat)
            # regional anomaly map ต่อภาพ: L2 norm ตามแกน channel ต่อ
            # ตำแหน่ง (h,w) — paper eq. 5
            error_map = torch.linalg.norm(feat - recon, dim=1)  # [B, H, W]

            for i in range(B):
                score = _aggregate_score(
                    error_map[i], self.cfg.SCORE_METHOD, self.cfg.SCORE_TOPK_RATIO)
                image_scores.append(score)

                pmap = error_map[i].view(1, 1, H, W)
                pmap = F.interpolate(pmap, size=self.cfg.IMAGE_SIZE,
                                      mode="bilinear", align_corners=False)
                pmap = _gaussian_smooth(pmap.squeeze().cpu().numpy(),
                                         self.cfg.HEATMAP_SIGMA)
                pixel_maps.append(pmap)

                orig_imgs.append(orig[i].permute(1, 2, 0).cpu().numpy().astype('float32'))
                preproc_imgs.append(preproc[i].permute(1, 2, 0).cpu().numpy().astype('float32'))

            y_true.extend([0 if lb == "normal" else 1 for lb in batch_labels])
            labels.extend(list(batch_labels))
            paths.extend(batch_paths)

        return ScoreResult(
            image_scores=np.array(image_scores, dtype=np.float64),
            y_true=np.array(y_true, dtype=np.int64),
            labels=labels,
            paths=paths,
            pixel_maps=np.stack(pixel_maps, axis=0),
            orig_imgs=np.stack(orig_imgs, axis=0),
            preproc_imgs=np.stack(preproc_imgs, axis=0),
        )


def _gaussian_smooth(arr: np.ndarray, sigma: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(arr, sigma=sigma)
