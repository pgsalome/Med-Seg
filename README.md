# Med-Seg

Modular medical image segmentation framework for MR/CT/PET workflows.

## What this repo includes

- Deterministic preprocessing with disk cache
- Mandatory N4 bias correction in brain pipeline
- QA PNG snapshots per preprocessing step (mid slice + lesion centroid)
- W&B logging for cache QA and training metrics
- Transfer-learning model hooks:
  - TRIAD PlainConvUNet encoder
  - TRIAD SwinB encoder
  - BrainIAC encoder wrapper (local clone)
- Patient-level cross-validation training script
- Optuna Bayesian hyperparameter tuning
- Preset-based configs for preprocessing and augmentation

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data registry format

Input registry is a JSON list of objects. Example:

```json
[
  {
    "patient_id": "010024",
    "site_group": "MOLAB",
    "IMG": "/path/image.nii.gz",
    "mask": "/path/mask.nii.gz",
    "label_id": 1,
    "test": 0
  }
]
```

Config keys controlling registry parsing:

- `data.image_key` (default `IMG`)
- `data.mask_key` (default `mask`)
- `data.id_keys` (default `["patient_id", "label_id"]`)

Registry loading now auto-normalizes common non-standard JSON layouts (for example `{"cases":[...]}` or `{patient_id: {...}}` maps) and key aliases.
If your pre-registry uses raw keys like `FU1_CT` and `RN`, build the new standardized registry first:

```bash
python scripts/normalize_registry.py \
  --cfg configs/experiment_triad_plain_preregistry.yaml \
  --in-image-key FU1_CT \
  --in-mask-key RN \
  --standardize \
  --out registries/registry.local.json
```

The pre-registry config (`configs/experiment_triad_plain_preregistry.yaml`) defines:

```yaml
data:
  image_key: "FU1_CT"
  mask_key: "RN"
  mask_id_key: "FU1_lesion_id"
```

If that `mask_id_key` field is missing in any row, the pipeline prints a warning and defaults `label_id=1`.
(`registry_local.mask_id_key` in `configs/base.yaml` can override this if needed.)

If you want a fixed constant `label_id` for all rows, use:

```bash
python scripts/normalize_registry.py \
  --cfg configs/experiment_triad_plain_preregistry.yaml \
  --in-image-key FU1_CT \
  --in-mask-key RN \
  --label-id-value 1 \
  --standardize \
  --out registries/registry.local.json
```

Training/cache scripts auto-build `registries/registry.local.json` from
`configs/experiment_triad_plain_preregistry.yaml` before loading data.

`configs/experiment_triad_plain_preregistry.yaml` is included as a ready pre-registry config.

To materialize only a normalized copy (without forcing standardized keys):

```bash
python scripts/normalize_registry.py --cfg configs/experiment_triad_plain.yaml
```

## Prepare cache + QA

```bash
python scripts/prepare_cache.py --cfg configs/experiment_triad_plain.yaml
```

Pipeline steps:

1. Resample to target spacing
2. N4 bias correction
3. Brain mask and crop
4. Robust clip and configurable normalization
5. Save `.npz` cache
6. Save QA PNGs for each step

Normalization mode is controlled in config via:

```yaml
preprocess:
  intensity:
    normalization_mode: "minmax"  # or "zscore"
```

## Train

TRIAD PlainConv encoder:

```bash
python scripts/train.py --cfg configs/experiment_triad_plain.yaml
```

TRIAD SwinB encoder:

```bash
python scripts/train.py --cfg configs/experiment_triad_swinb.yaml
```

BrainIAC:

1. Clone BrainIAC locally.
2. Set `model.brainiac.repo_root` and `model.brainiac.checkpoint_path` in config.
3. Run:

```bash
python scripts/train.py --cfg configs/experiment_brainiac.yaml
```

Patient-level cross-validation:

```bash
python scripts/train_cv.py --cfg configs/experiment_triad_plain.yaml --folds 5
```

## Optuna tuning

```bash
python scripts/optuna_tune.py --cfg configs/experiment_triad_plain.yaml --trials 25
```

## Extend to new organs/modalities

Add new preset files, then compose a new experiment config with `inherits`.
