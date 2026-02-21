# Registries

Put your dataset registry JSON files here.

- Preferred standardized schema is `patient_id`, `IMG`, `mask`, `label_id`.
- Use `registry.example.json` as the template.
- Create your local file as `registry.local.json` (this is ignored by git).
- If your raw registry uses other keys (for example `FU1_CT` + `RN`), convert it with:
  `python scripts/normalize_registry.py --cfg configs/experiment_triad_plain_preregistry.yaml --in-image-key FU1_CT --in-mask-key RN --standardize --out registries/registry.local.json`
- If `label_id` is constant, use `--label-id-value 1` instead of `--label-id-field ...`.
- Source field for IDs is configured in `configs/experiment_triad_plain_preregistry.yaml` as `data.mask_id_key`.
- If `data.mask_id_key` is missing per-row, code warns and defaults `label_id=1`.
- `prepare_cache.py`, `train.py`, and `train_cv.py` auto-generate `registry.local.json` before loading data.

The default config points to:

- `data.registry_json: "registries/registry.local.json"`
