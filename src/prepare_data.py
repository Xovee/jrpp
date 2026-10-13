"""Prepare JRPP's fixed inputs directly from a published SKAPP dataset ZIP."""

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from utils.data_contract import FEATURES, SPLITS, digest, ids_digest, validate_arrays


def prepare(dataset, archive, output):
    """Copy shared inputs exactly; never resplit, transform labels or overwrite data."""
    archive, output = Path(archive), Path(output)
    if dataset not in ("icip", "smpd", "instagram"):
        raise ValueError("Unknown dataset")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    fields = ["mean_views"] if dataset == "icip" else []
    manifest = {
        "format": "popularity-fixed-splits-v1",
        "dataset": dataset,
        "meta_fields": fields,
        "split_policy": "fixed-reference-order-no-resplitting",
        "label_policy": "published-target-float32-no-additional-transform",
        "feature_policy": "exact-reference-inputs",
        "reference_archive_sha256": digest(archive),
        "splits": {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Only publish an output directory after every split has passed validation.
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=output.parent) as staging:
        staging = Path(staging)
        seen = set()
        with zipfile.ZipFile(archive) as reference:
            reference_manifest = json.loads(reference.read(f"{dataset}/dataset.json"))
            if (
                reference_manifest.get("dataset") != dataset
                or reference_manifest.get("format_version") != 1
            ):
                raise ValueError("Wrong reference dataset or format")
            if set(reference_manifest["splits"]) != set(SPLITS):
                raise ValueError("Expected train, valid and test reference splits")
            for split in SPLITS:
                temporary = staging / "reference.npz"
                with reference.open(f"{dataset}/{split}.npz") as src, temporary.open("wb") as dst:
                    shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
                entry = reference_manifest["splits"][split]
                if (
                    temporary.stat().st_size != entry["bytes"]
                    or digest(temporary) != entry["sha256"]
                ):
                    raise ValueError(f"{split}: reference checksum mismatch")
                with np.load(temporary, allow_pickle=False) as ref:
                    required = ("image_id", "user_id", "label", *FEATURES, *fields)
                    missing = set(required).difference(ref.files)
                    if missing:
                        raise ValueError(
                            f"{split}: missing inputs {sorted(missing)}; use the current SKAPP package"
                        )
                    arrays = {key: ref[key] for key in required}
                # Match Torch scalar precision while keeping feature arrays unchanged.
                for key in ("label", *fields):
                    if arrays[key].dtype.kind not in "fiu":
                        raise ValueError(f"{split}: nonnumeric {key}")
                    arrays[key] = arrays[key].astype(np.float32)
                temporary.unlink()
                validate_arrays(arrays, fields)
                ids = arrays["image_id"].tolist()
                if (
                    len(ids) != entry["rows"]
                    or hashlib.sha256("\n".join(ids).encode()).hexdigest() != entry["ids_sha256"]
                ):
                    raise ValueError(f"{split}: reference sample order mismatch")
                if seen.intersection(ids):
                    raise ValueError("Overlapping reference splits")
                seen.update(ids)
                target = staging / f"{split}.npz"
                np.savez(target, **arrays)
                manifest["splits"][split] = {
                    "file": target.name,
                    "rows": len(ids),
                    "bytes": target.stat().st_size,
                    "sha256": digest(target),
                    "ids_sha256": ids_digest(ids),
                    "shared_inputs_exactly_match_reference": True,
                }
                print(f"{dataset}/{split}: {len(ids)} samples verified", flush=True)
        (staging / "dataset.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        staging.rename(output)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("icip", "smpd", "instagram"))
    parser.add_argument("--archive", required=True, help="Published skapp-<dataset>.zip")
    parser.add_argument("--output", required=True, help="New dataset directory, e.g. data/icip")
    args = parser.parse_args()
    prepare(args.dataset, args.archive, args.output)
