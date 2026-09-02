# PCB-Anomaly-Baselines-DFR

Standalone baseline repo implementing **DFR** (Yang, Shi & Qi, *"DFR: Deep
Feature Reconstruction for Unsupervised Anomaly Segmentation"*,
arXiv:2012.07122) for the PCB defect-inspection thesis project — a
literature/SOTA comparison row alongside `Anomaly-Detection-THESIS`
(EXPERIMENT 0, ConvNeXt+AE) and `PCB-Anomaly-Baselines-PatchCore`.

**This version implements DFR as close to the original paper as
practical, by default** — VGG19 backbone, nearest-neighbor align +
mean-filter aggregation (paper eq. 1-2), no feature normalization,
reflection padding (paper §IV-C.2) — rather than adapting it to match
EXPERIMENT 0's ConvNeXt/z-score conventions. This is deliberate: the
thesis hasn't decided yet which backbone/normalization it will ultimately
use, so this repo keeps a literature-faithful DFR available as a clean
reference point, uncontaminated by the thesis's own design choices. Every
one of these choices is still a plain `Config` field — switch to the
ConvNeXt-matched variant any time (see below) without touching code.

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
5. Outputs: `SAVE_PATH` (default `save/logs`) for `.npz`/`.json`/`.csv`/
   `history.json`; `OUTPUT_PATH` (default `save/results`) for all `.png`
   plots including `training_history.png`.

## Switching to the ConvNeXt-matched variant

If/when the thesis settles on comparing every baseline with the same
backbone as EXPERIMENT 0, replace these 6 `OVERRIDES` keys in `RUN.py`:

```python
BACKBONE="convnext_tiny",
FEATURE_LAYERS=("features.3", "features.5"),
DFR_ALIGN_MODE="bilinear",
NORMALIZE_FEATURES=True,
# DFR_AGG_KERNEL / DFR_AGG_STRIDE / DFR_REFLECTION_PADDING can stay at
# their defaults either way.
```

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
