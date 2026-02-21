import json
import os
import sys
from typing import Any, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from medseg.config import load_config
from medseg.io_registry import normalize_registry_file
from medseg.registry_build import build_standard_registry_from_source_cfg


def main(
    cfg_path: str,
    output_path: Optional[str] = None,
    input_image_key: Optional[str] = None,
    input_mask_key: Optional[str] = None,
    label_id_field: Optional[str] = None,
    label_id_value: Any = 1,
    label_id_key: str = "label_id",
    standardize: bool = False,
):
    cfg = load_config(cfg_path)
    if input_image_key:
        cfg["data"]["image_key"] = input_image_key
    if input_mask_key:
        cfg["data"]["mask_key"] = input_mask_key

    registry_path = cfg["data"]["registry_json"]
    if standardize:
        stats = build_standard_registry_from_source_cfg(
            source_cfg_path=cfg_path,
            output_path=output_path or f"{os.path.splitext(registry_path)[0]}.standard.json",
            input_image_key=input_image_key,
            input_mask_key=input_mask_key,
            label_id_field=label_id_field,
            label_id_default=label_id_value,
            label_id_key=label_id_key,
        )
        print(f"Normalized rows: {stats['rows']}")
        print(f"Saved: {stats['output_path']}")
        return

    rows = normalize_registry_file(registry_path, cfg)
    if output_path is None:
        base, _ = os.path.splitext(registry_path)
        output_path = f"{base}.normalized.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2)

    print(f"Normalized rows: {len(rows)}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--in-image-key", default=None)
    parser.add_argument("--in-mask-key", default=None)
    parser.add_argument("--label-id-field", default=None)
    parser.add_argument("--label-id-value", default=1)
    parser.add_argument("--label-id-key", default="label_id")
    # Backward-compatible aliases.
    parser.add_argument("--mask-id-field", default=None)
    parser.add_argument("--mask-id-value", default=None)
    parser.add_argument("--mask-id-key", default=None)
    parser.add_argument("--standardize", action="store_true")
    args = parser.parse_args()

    effective_label_id_field = args.label_id_field or args.mask_id_field
    effective_label_id_value = (
        args.label_id_value if args.mask_id_value in (None, "") else args.mask_id_value
    )
    effective_label_id_key = args.label_id_key if args.mask_id_key in (None, "") else args.mask_id_key

    main(
        args.cfg,
        output_path=args.out,
        input_image_key=args.in_image_key,
        input_mask_key=args.in_mask_key,
        label_id_field=effective_label_id_field,
        label_id_value=effective_label_id_value,
        label_id_key=effective_label_id_key,
        standardize=args.standardize,
    )
