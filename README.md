# PCB-Anomaly-Baselines-DFR

Standalone baseline repo implementing **DFR** (Yang, Shi & Qi, *"DFR: Deep
Feature Reconstruction for Unsupervised Anomaly Segmentation"*,
arXiv:2012.07122) for the PCB defect-inspection thesis project — a
literature/SOTA comparison row alongside `Anomaly-Detection-THESIS`
(EXPERIMENT 0, ConvNeXt+AE) and `PCB-Anomaly-Baselines-PatchCore`.

**`RUN.py`'s active config is now ConvNeXt-tiny Stage2+3+4** — the same
backbone family as EXPERIMENT 0/PatchCore/PaDiM, chosen deliberately so
this baseline's numbers isolate DFR's *method* (CAE design, reconstruction
scoring) rather than being confounded by backbone differences (VGG19 is a
much older architecture than ConvNeXt; if DFR-on-VGG19 loses to
ConvNeXt-based EXPERIMENT 0, that comparison wouldn't say anything about
the *method*). `config/config.py`'s raw dataclass defaults stay VGG19 —
paper-faithful (nearest-neighbor align, mean-filter aggregation per paper
eq. 1-2, no feature normalization, reflection padding per paper §IV-C.2) —
as a literature-reproduction reference, separate from what `RUN.py`
actually runs. Every one of these choices is a plain `Config` field —
switch between them any time (see below) without touching code.

## Quickstart

1. Edit `RUN.py` → `OVERRIDES`: set `DATA_ROOT`. For a single comparison
   run against `Anomaly-Detection-THESIS`/`PCB-Anomaly-Baselines-PatchCore`,
   point `SPLIT_CACHE_PATH` at their **same** `split_assignment.csv` so
   train/val/test membership matches exactly. For multi-seed runs via
   `RUN_MULTI_SEED.py`, the default layout instead gives each seed its own
   `log`/`table`/`split` (see "Multi-seed folder layout" below) — a
   deliberate trade-off, not an oversight.
2. `pip install -r requirements.txt`
3. **Before trusting any result**, run `python tests/heatmap_sanity_check.py`
   with `PRETRAINED=True` (needs internet) and confirm the heatmap
   localizes the synthetic patch correctly — see "Heatmap sanity check
   finding" below for why this matters.
4. `python RUN.py` — fit → score → save → visualize → cost-aware sweep.
5. Outputs: `SAVE_PATH` (default `save/SEED 42/log`) for `.npz`/`.json`/
   `.csv`/`history.json`; `OUTPUT_PATH` (default `save/SEED 42/table`) for
   all `.png` plots including `training_history.png`.

## ConvNeXt (active default) vs. VGG19 (paper-faithful) — when to use which

| Use case | Backbone | Why |
|---|---|---|
| Main comparison table vs. EXPERIMENT 0/PatchCore/PaDiM | **ConvNeXt-tiny Stage2+3+4** (`RUN.py` default now) | Same backbone family across every baseline — isolates the *algorithm* difference, matching how PatchCore/PaDiM/SimpleNet papers compare against each other |
| Sanity-check that this reimplementation reproduces the paper's own numbers | **VGG19 front-12** (`config/config.py` raw default) | Confirms the code is a faithful DFR implementation, not a broken one — useful as an appendix/footnote, not the headline comparison |

Module names for ConvNeXt-tiny were verified against the actual
`torchvision.models.convnext_tiny()` via forward hooks (not guessed):
`features.3`=Stage2 (192ch, 28×28 @ 224×224 input), `features.5`=Stage3
(384ch, 14×14), `features.7`=Stage4 (768ch, 7×7) — `c_in = 1344` combined,
actually lighter than VGG19 front-12's `c_in = 3456`.

To switch back to the VGG19 variant in `RUN.py`, see the commented block
right below the active `OVERRIDES` (swap the 6 keys `BACKBONE` through
`NORMALIZE_FEATURES`).

## Files reused verbatim from PCB-Anomaly-Baselines-PatchCore / Anomaly-Detection-THESIS

Byte-identical (verified with `diff`) — **do not edit without warning
first**:

- `src/data/dataset.py` — group-based 70/15/15 split, defect never enters
  train (hard-asserted), split caching
- `src/evaluate.py` — `compute_metrics`, `select_percentile_threshold`,
  `compute_naive_baseline_metrics`
- `src/models/base.py` — `BaseAnomalyModel`, `ScoreResult`
- `src/cost_aware.py` — cost-aware threshold sweep (fully method-agnostic)
- `src/visual.py` — reused almost entirely; `plot_training_history` is
  replaced with a train-loss-only version since DFR has no `val_loader`
  during `fit()` (see "Differences from the main repo" below)

## What's DFR-specific

- `src/models/dfr.py` — frozen backbone feature extraction → **align**
  (nearest-neighbor resize to input image size, paper eq. 1) → **aggregate**
  (mean-filter/avg-pool, `DFR_AGG_KERNEL`/`DFR_AGG_STRIDE`, paper eq. 2) →
  **concatenate** (paper eq. 3) → 6-layer 1×1-conv CAE (Appendix B) → L2
  reconstruction-error map → mean/max/topk image-level aggregation
  (`SCORE_METHOD`)
- `config/config.py` — DFR-only fields: `DFR_EPOCHS`, `DFR_LR`,
  `DFR_ALIGN_MODE`, `DFR_AGG_KERNEL`, `DFR_AGG_STRIDE`,
  `DFR_REFLECTION_PADDING`, `DFR_LATENT_DIM`, `DFR_PCA_VARIANCE_RATIO`,
  `NORMALIZE_FEATURES`, `SCORE_METHOD`, ...
- `scripts/run_dfr.py`, `scripts/visualize_dfr.py`,
  `scripts/run_cost_aware_dfr.py`, `RUN.py`, `RUN_MULTI_SEED.py` — same
  3-step pattern as the PatchCore repo
- `tests/smoke_test.py` — offline pipeline-mechanics check (shapes, no
  crashes) on tiny dummy data
- `tests/heatmap_sanity_check.py` — **content-level** check: does the
  produced heatmap actually track a known synthetic defect location, or
  is it just scattered/structured-but-wrong? See finding below.

## Chunked feature extraction — CUDA OOM fix

Reported on real hardware (Colab, 22GB GPU) during a `RUN_MULTI_SEED.py`
run: `CUDA out of memory` inside `_regional_feature_map`'s `F.interpolate`
call. Root cause and fix:

1. **Per-batch cause**: the align step (paper eq. 1) resizes *every* scale
   to the full input image resolution before pooling it back down. For a
   deep VGG19 layer (512 channels) at `BATCH_SIZE=32` and `IMAGE_SIZE=224`,
   that single intermediate tensor is `32×512×224×224×4 bytes ≈ 13 GB` —
   for one of twelve scales, before aggregation shrinks it back down.
2. **Cross-seed cause**: `RUN_MULTI_SEED.py` runs many seeds sequentially
   in one process. Without explicit cleanup, PyTorch's CUDA caching
   allocator accumulates "reserved but unallocated" memory across seeds —
   matching the reported error (`20.31 GiB memory in use ... 8.51 GiB is
   reserved by PyTorch but unallocated`).

Fixes, both on by default:

- **`DFR_FEATURE_CHUNK_SIZE`** (default `8`): `_regional_feature_map`
  processes the batch in sub-chunks of this size instead of all at once,
  bounding peak memory for the align step regardless of `BATCH_SIZE`.
  Verified numerically equivalent to the unchunked path up to
  floating-point rounding (~1e-6 relative — ordinary batch-size-dependent
  conv/BatchNorm non-associativity, not an algorithmic approximation; see
  the diagnostic in this repo's development history). Lower it (e.g. `4`
  or `2`) if OOM persists — this does **not** require lowering
  `BATCH_SIZE` itself, which would otherwise also perturb the Adam
  gradient estimate.
- **Explicit cleanup**: `DFR.fit()`/`DFR.score()` call
  `torch.cuda.empty_cache()` after use; `run_dfr.run()` additionally
  `del`s the model and calls `gc.collect()` before returning;
  `RUN_MULTI_SEED.py` does the same after every seed (success *or*
  failure, via a `finally` block) and sets
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` by default (as
  PyTorch's own OOM message suggests) unless already set.

**Caveat**: this development sandbox has no GPU
(`torch.cuda.is_available()` is `False` here), so only the CPU-side
correctness could be verified directly here — the chunking logic is
confirmed numerically equivalent to the unchunked path (see above), and
the code review/compile checks all pass, but the *actual* OOM avoidance on
real GPU hardware could not be executed and observed in this sandbox. If
`DFR_FEATURE_CHUNK_SIZE=8` still OOMs on your hardware, lower it further
(down to `1` if needed) — memory scales roughly linearly with this value.

## Differences from the original DFR paper (stated explicitly)

These describe `config/config.py`'s raw `Config` dataclass defaults (the
paper-faithful reference) — **not** what `RUN.py` actually runs by
default anymore (`RUN.py` now uses ConvNeXt Stage2+3+4, see "ConvNeXt vs.
VGG19" above). Instantiating `Config()` with no overrides still gives you
this literature-faithful setup.

1. **Backbone default**: VGG19 front-12 layers (`VGG19_FRONT_12_LAYERS` /
   the first 12 entries of `VGG19_ALL_16_LAYERS` in `src/models/dfr.py`,
   verified against the actual `torchvision.models.vgg19().features`
   module names, not guessed) instead of all 16. The paper's own §IV-C.1
   finds f{1:12} within ~0.01 ROC-AUC of f{1:16} at meaningfully lower
   compute — set `FEATURE_LAYERS=VGG19_ALL_16_LAYERS` for full fidelity to
   the paper's headline numbers if compute allows (c_o balloons to ~5500
   channels at native VGG19 depth).
2. **Regional feature generator now matches paper eq. 1-2 exactly**: align
   via nearest-neighbor to the *input image's* spatial size (not the
   finest backbone feature map, which an earlier version of this repo
   mistakenly used), then mean-filter aggregate with `DFR_AGG_KERNEL`/
   `DFR_AGG_STRIDE` (default 4/4, matching the paper's MVTec AD setup).
3. **`NORMALIZE_FEATURES=False` by default** — the paper does not
   z-score-normalize features before the CAE. `True` is available for
   `Anomaly-Detection-THESIS` convention parity.
4. **Reflection padding is implemented and on by default**
   (`DFR_REFLECTION_PADDING=True`) — see the dedicated section below.
5. **Latent dimension**: same idea as the paper (PCA, 90% variance,
   `DFR_PCA_VARIANCE_RATIO`), estimated from a random subset
   (`DFR_PCA_SAMPLE_SIZE`) rather than the whole training set, for memory
   reasons. Set `DFR_LATENT_DIM` to skip PCA entirely.

## Reflection padding — a real bug found and fixed during development

The paper's §IV-C.2 ("Boundary Effects") reports that zero-padding in the
backbone causes spurious high error near image edges, and proposes
reflection padding as the fix (~1% average ROC-AUC/PRO-AUC improvement).
While building this repo I verified this is not a minor effect:

- Diagnostic: ran two **completely unrelated random images** through the
  frozen backbone and compared their per-position feature-magnitude maps.
  With zero-padding, correlation was **0.95** — i.e. the "heatmap" was
  driven almost entirely by *position*, not image content. Switching every
  `Conv2d.padding_mode` from `'zeros'` to `'reflect'` (matching the paper's
  own fix) dropped this to **0.04**.
- `src/models/dfr.py::_FeatureExtractor` now does this switch automatically
  whenever `cfg.DFR_REFLECTION_PADDING=True` (default), for whichever
  `BACKBONE` is configured.

### Reflection padding + small IMAGE_SIZE — a real edge case

Reflection padding requires `padding < spatial size` for every convolution
(zero-padding has no such constraint). This can break for **deep** layers
combined with a **small** `IMAGE_SIZE`: e.g. ConvNeXt Stage4
(`features.7`) downsamples by 32× from the input, so at `IMAGE_SIZE=96` it
shrinks to 3×3 — too small for ConvNeXt's 7×7 depthwise convs
(`padding=3`, which needs spatial ≥ 4). This was caught during development
(a quick integration test using a small dummy image size hit it
immediately) and is now a clear, actionable error instead of a raw
PyTorch stack trace — `_FeatureExtractor.forward()` catches this specific
failure and tells you exactly what to do:

```
Reflection padding พังเพราะ IMAGE_SIZE เล็กเกินไปสำหรับ BACKBONE=ConvNeXt +
FEATURE_LAYERS=(...) — ... แก้ได้ 2 ทาง: (1) เพิ่ม IMAGE_SIZE ... หรือ
(2) ตั้ง DFR_REFLECTION_PADDING=False ...
```

**Verified safe**: the real default (`IMAGE_SIZE=224`, ConvNeXt
Stage2+3+4) does **not** hit this — Stage4 lands at 7×7, and `3 < 7`
passes. Confirmed via a full `run_dfr.run()` pass at `IMAGE_SIZE=224` with
the exact `RUN.py` `OVERRIDES` (PCA landed on 435 latent dims from 1344
input channels; `AUC=1.0` on both val/test on the dummy data). Only
relevant if you deliberately lower `IMAGE_SIZE` below roughly 112–128 for
speed while using deep layers like ConvNeXt Stage4 or VGG19's later
layers — if you do, either raise `IMAGE_SIZE` back up or set
`DFR_REFLECTION_PADDING=False` for that run.

## Heatmap sanity check finding — read before trusting real results

`tests/heatmap_sanity_check.py` builds a dummy dataset where every
"defect" image has a bright patch pasted at a **randomized-per-image**
location, then checks whether the produced heatmap's peak actually tracks
that location (a fixed/repeated patch location would make this check
meaningless — an earlier version of this test made exactly that mistake
and was corrected).

**Result with `PRETRAINED=False`** (this sandbox has no internet access to
download real ImageNet weights, so this is the only mode testable here):
the check **fails**. Investigation:

1. Reflection padding fixes the *content-independent* positional artifact
   (confirmed above) — heatmaps on plain **normal** images now look
   properly textured/content-varying, not a fixed vignette.
2. But on **defect** images, a large, high-error region still appears —
   and critically, **its location does not move when the patch's location
   is randomized per image** (verified visually: patch top-right → hot
   region unchanged; patch top-left → hot region still unchanged, on the
   opposite side of the image from the patch). This rules out the patch
   itself, and rules out receptive-field spread around the patch, as the
   explanation.
3. More training epochs (15→60) and a larger latent dimension (4→16) made
   no difference — ruling out simple undertraining.

**Most likely explanation**: an *untrained, random-init* backbone has none
of a trained network's Lipschitz-type stability, so a strong local
perturbation (a stark bright patch) can trigger a non-local, essentially
chaotic response in deeper layers — a known failure mode of random-weight
CNNs that trained weights don't exhibit. This is consistent with the
pipeline mechanics being entirely correct (`AUC=1.0` on the *aggregate*
image-level score throughout every test run here — the model still ranks
defect images as more anomalous, it just doesn't localize *within* a
defect image reliably at the pixel level when the backbone is untrained).

**This is not a confirmed bug in `dataset.py`, `evaluate.py`, the
regional-feature-generator, or the CAE** — every other check (shapes,
calibration, metrics, cost-aware sweep, multi-scale PCA path) passed
cleanly. But it **is** a real, demonstrated limitation of running with
`PRETRAINED=False`, which is exactly why that flag is documented
everywhere in this repo as smoke-test-only.

**Action required before trusting any real DFR heatmap from this repo**:
run `python tests/heatmap_sanity_check.py` yourself with `PRETRAINED=True`
(on a machine with internet access to download the ImageNet weights) and
confirm the peak-in-patch hit rate is well above chance level before
drawing conclusions from real-dataset heatmaps. If it still fails with real
weights, that *would* indicate a genuine bug worth reopening this
investigation for — but that hasn't been observed here, only under the
random-init constraint this sandbox is stuck with.

## Differences from the main repo (`Anomaly-Detection-THESIS`, EXPERIMENT 0)

`DFR.fit()` implements `BaseAnomalyModel.fit(self, normal_loader)` — only a
normal-only train loader, no `val_loader`. Every baseline in this family
(PatchCore, PaDiM, DRAEM, SimpleNet, RD4AD, DFR) shares this interface, so
DFR's CAE training here **cannot** replicate EXPERIMENT 0's `AE_MONITOR`
early-stopping/checkpoint-selection behavior:

- Training runs a **fixed** `DFR_EPOCHS` schedule with `StepLR`, no early
  stopping, no best-checkpoint selection by validation metric.
- `history.json`/`training_history.png` have only `train_loss` (normal
  only) — no `val_loss`, `val_loss_normal`, or `val_auroc` curves.

## Multi-seed folder layout — per-seed split (deliberate, not a bug)

`RUN.py`'s default layout is `save/SEED {n}/log`, `save/SEED {n}/table`,
`save/SEED {n}/split` — every seed gets its **own** `split_assignment.csv`
too (`RUN_MULTI_SEED.py`'s `TEMPLATE_KEYS` includes `SPLIT_CACHE_PATH`),
not a single split shared across all seeds.

**Trade-off worth knowing**: with a per-seed split, variance measured
across seeds now mixes two sources — (1) training randomness (CAE weight
init/batch order) and (2) different train/val/test membership per seed
(since the split itself is recomputed per seed). This means per-seed
results can no longer be compared image-for-image against
`Anomaly-Detection-THESIS`/`PatchCore` (which use one fixed split), and a
seed-to-seed metric spread reflects "how sensitive is DFR to both training
randomness *and* which images landed in train vs. val/test" rather than
training randomness alone. If instead you want to isolate training
randomness only (comparable to a fixed-split baseline), remove
`'SPLIT_CACHE_PATH'` from `TEMPLATE_KEYS` in `RUN_MULTI_SEED.py` and drop
`"SEED 42"` from `SPLIT_CACHE_PATH` in `RUN.py` — every seed then shares
one split file, computed once.

## Differences from `PCB-Anomaly-Baselines-PatchCore` (bugs found + fixed here)

While building this repo I cloned and read the actual PatchCore repo
source (not just its documentation) and found two real issues, fixed here
rather than copied:

1. **`io_utils.py` hardcodes `"method": "PatchCore"` and a
   `"coreset_ratio": cfg.CORESET_RATIO"` field** inside
   `save_final_results()`. Fixed via `cfg.METHOD_NAME` + an optional
   `extra_fields: dict` parameter — both backward-compatible, so this
   version could be backported to `PCB-Anomaly-Baselines-PatchCore`.
2. **`RUN_MULTI_SEED.py` requires a `"SEED {n}"` marker** that the
   PatchCore repo's actual `RUN.py` never embeds — running it as-is raises
   `ValueError` immediately. This repo's `RUN.py` embeds `"SEED 42"` in
   every templated path (see "Multi-seed folder layout" above).

Also added (PatchCore has no training loop to need these):
`src/io_utils.py::save_training_history()`, and
`src/output_docs.py::write_output_path_readme()` is actually wired up and
called from `visualize_dfr.py` (in the PatchCore repo this function exists
but is never called, so `OUTPUT_PATH/README.md` never gets generated
there).

## Labeled confusion matrix in final_results.json

`final_results_{split}.json`'s `"confusion_matrix"` field used to be a bare
2×2 array (`metrics["cm"].tolist()`) — you had to already know sklearn's
`confusion_matrix(y_true, y_pred, labels=[0,1])` convention
(`[[tn, fp], [fn, tp]]`) to read it. It's now a labeled dict, computed in
`src/io_utils.py::_labeled_confusion_matrix()` (not in `src/evaluate.py`,
which must stay byte-identical to the other repos):

```json
"confusion_matrix": {
  "tt": 4, "tf": 0, "ft": 1, "ff": 2,
  "definitions": {
    "tt": "actual=anomaly(defect), predicted=anomaly -> caught defect (true positive)",
    "tf": "actual=anomaly(defect), predicted=normal  -> escaped/missed defect (false negative)",
    "ft": "actual=normal(good),    predicted=anomaly -> false alarm (false positive)",
    "ff": "actual=normal(good),    predicted=normal  -> correctly auto-cleared (true negative)"
  },
  "sklearn_raw_cm": [[2, 1], [0, 4]],
  "sklearn_raw_cm_note": "... tn==ff, fp==ft, fn==tf, tp==tt ..."
}
```

`tt`/`tf`/`ft`/`ff` match the same naming already used in this project's
`naive_baselines` section (`src/evaluate.py::compute_metrics_from_predictions`)
— first letter = actual label, second = predicted label, `T`=anomaly/defect,
`F`=normal/good. Cross-checked against the sibling `metrics` fields:
`escape_rate = tf/(tt+tf)`, `auto_clear_rate = ff/n`,
`residual_fcr = ft/(tt+ft)`.

## MVTec AD reproduction (separate pipeline — not the PCB thesis one)

`scripts/run_dfr.py`/`RUN.py` (everything above) target the **PCB thesis
dataset** — group-based random split, image-level classification metrics.
Reproducing the DFR **paper's own** published numbers (Table II/III)
needs a genuinely different pipeline, added as a **parallel, additive**
set of files — nothing above was touched:

| New file | Purpose |
|---|---|
| `config/mvtec_config.py` | `MVTecConfig` — separate dataclass, paper-exact defaults (VGG19, `IMAGE_SIZE=(256,256)`, `BATCH_SIZE=4`, `DFR_EPOCHS=700`, no feature normalization, reflection padding on) |
| `src/data/mvtec_dataset.py` | Parses MVTec's official `train/`/`test/`/`ground_truth/` layout; reuses `AnomalyDataset`/`build_transforms`/`make_loader` from the verbatim `dataset.py` directly (same preprocessing as every other baseline) |
| `src/mvtec_evaluate.py` | Pixel-level ROC-AUC + region-level PRO-AUC (paper §IV-A.5's actual protocol — connected-component overlap, normalized to ≤30% average pixel-FPR) |
| `scripts/run_dfr_mvtec.py` | Loops over categories, trains+scores DFR per category, prints/saves a Table II/III-style summary |
| `tests/mvtec_metric_unit_test.py` | Validates `pixel_roc_auc`/`pro_auc` against synthetic ground truth with known answers (perfect prediction → ~1.0, random → ~0.5/low, all-normal → NaN not a crash, partial overlap → sane middle value) — **run this before trusting the metric on anything else** |
| `tests/mvtec_pipeline_test.py` | Full structural test: builds a fake MVTec-formatted folder tree (`train/good`, `test/good`, `test/{defect}`, `ground_truth/{defect}/*_mask.png`), runs the entire pipeline through it |

### Why `dataset.py`/`evaluate.py` weren't extended instead

MVTec's train/test split is **fixed by the dataset**, not computed — using
`dataset.py`'s group-based random-split logic would silently produce a
non-standard split incomparable to any published number, defeating the
entire point of "reproducing the paper." Its headline metrics are also
fundamentally different (pixel/region-level, not image classification), so
`evaluate.py`'s `compute_metrics()` doesn't apply either. Keeping this as
a separate, additive path avoids two bad options: bolting MVTec-specific
branches onto the verbatim PCB files (breaking their "byte-identical
across repos" guarantee), or silently reusing them in a way that produces
plausible-looking but methodologically wrong numbers.

### What's actually verified vs. not

- **Metric correctness**: `tests/mvtec_metric_unit_test.py` passes —
  perfect prediction gives ROC-AUC=1.000/PRO-AUC=1.000, random prediction
  gives ROC-AUC≈0.49/PRO-AUC=0.15 (low), all-normal masks return `NaN`
  without crashing, partial region overlap gives PRO-AUC=0.75 (a sane
  middle value, not a degenerate 0 or 1).
- **Pipeline mechanics**: `tests/mvtec_pipeline_test.py` passes on a
  synthetic fake-MVTec folder tree — folder parsing, mask loading/lookup,
  `DFR.fit()`/`.score()`, and both metrics all run correctly together
  (ROC-AUC≈0.99, PRO-AUC≈0.96 on an easy synthetic anomaly, with
  `PRETRAINED=False` and only 3 epochs).
- **NOT verified: real MVTec AD numbers.** This sandbox has no network
  access to any MVTec AD download source (official site or mirrors) — none
  of the domains needed are in the allowed list. **You must download and
  extract MVTec AD yourself** (official page:
  `https://www.mvtec.com/company/research/datasets/mvtec-ad`, ~4.9 GB) and
  point `MVTEC_ROOT` in `scripts/run_dfr_mvtec.py`'s `main()` call at the
  extracted folder, then run:
  ```bash
  python -c "
  from config.mvtec_config import MVTecConfig
  from scripts.run_dfr_mvtec import main
  main(MVTecConfig(MVTEC_ROOT='/path/to/mvtec_anomaly_detection'))
  "
  ```
  With the paper's own settings (`DFR_EPOCHS=700`, all 15 categories),
  expect this to take hours per category on GPU — start with
  `CATEGORIES=("carpet",)` to sanity-check one category first. Compare
  the resulting `roc_auc`/`pro_auc` against paper Table II/III's "Ours
  f{1:12}" column (this repo's `FEATURE_LAYERS` default) before trusting
  a full 15-category run.

## Verified end-to-end (dummy data, offline backbone)

- `tests/smoke_test.py` (VGG19-analog light backbone, 2 epochs): pipeline
  mechanics — shapes, `.npz` schema, `history.json` — all correct.
- Full integration run (`run_dfr.run()` → `visualize_dfr.visualize()` →
  `run_cost_aware_dfr.main()`) with the actual paper-faithful `RUN.py`
  defaults, including the real VGG19 front-12 backbone and the PCA
  auto-latent-dim path (landed on 887 components from 3456 input channels
  at 90% variance): completed without errors, correct output schema.
- `tests/heatmap_sanity_check.py`: pipeline mechanics pass; **heatmap
  content-localization does not**, for the documented random-init-backbone
  reason above — re-validate with `PRETRAINED=True` before trusting real
  heatmaps.

None of this confirms real-dataset metrics are meaningful. Before trusting
any number from this repo: set `PRETRAINED=True` (the `Config` default),
point `SPLIT_CACHE_PATH` at the shared split, run on the real dataset, and
re-run `tests/heatmap_sanity_check.py` with real weights first.
