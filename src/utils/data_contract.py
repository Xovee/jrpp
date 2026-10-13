"""Model-independent, fixed-split NumPy data contract."""

import hashlib
import json
from pathlib import Path

import numpy as np

SPLITS = ("train", "valid", "test")
FEATURES = ("cls_vec", "merged_text_vec")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def ids_digest(ids):
    return hashlib.sha256(
        json.dumps(list(ids), ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def validate_arrays(arrays, meta_fields):
    ids = arrays["image_id"]
    n = len(ids)
    if not n or ids.shape != (n,) or ids.dtype.kind not in "US":
        raise ValueError("Expected nonempty one-dimensional string image IDs")
    if len(set(ids.tolist())) != n or any(not x for x in ids):
        raise ValueError("Duplicate or empty image IDs")
    users = arrays["user_id"]
    if users.shape != (n,) or users.dtype.kind not in "US" or any(not x for x in users):
        raise ValueError("Invalid user IDs")
    for field in ("label", *FEATURES, *meta_fields):
        a = arrays[field]
        shape = (n, 768) if field in FEATURES else (n,)
        if a.shape != shape or a.dtype != np.float32 or not np.isfinite(a).all():
            raise ValueError(f"Invalid {field}: expected finite float32 {shape}")

    if "mean_views" in meta_fields and (arrays["mean_views"] < 0).any():
        raise ValueError("mean_views must be nonnegative")


def load_contract(source, data_name, meta_fields):
    source = Path(source)
    manifest = json.loads((source / "dataset.json").read_text(encoding="utf-8"))
    if (
        manifest.get("format") != "popularity-fixed-splits-v1"
        or manifest.get("dataset") != data_name
    ):
        raise ValueError("Dataset format or identity mismatch")
    if manifest.get("meta_fields") != list(meta_fields):
        raise ValueError("Configured metadata fields differ from prepared dataset")
    if set(manifest["splits"]) != set(SPLITS):
        raise ValueError("Expected train, valid and test splits")
    result = {}
    seen = set()
    for split in SPLITS:
        entry = manifest["splits"][split]
        if entry["file"] != f"{split}.npz":
            raise ValueError("Invalid split filename")
        path = source / entry["file"]
        if path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            raise ValueError(f"{split} checksum mismatch")
        with np.load(path, allow_pickle=False) as data:
            arrays = {k: data[k] for k in ("image_id", "user_id", "label", *FEATURES, *meta_fields)}
        validate_arrays(arrays, meta_fields)
        ids = arrays["image_id"].tolist()
        if len(ids) != entry["rows"] or ids_digest(ids) != entry["ids_sha256"]:
            raise ValueError(f"{split} sample order mismatch")
        if seen.intersection(ids):
            raise ValueError("Overlapping dataset splits")
        seen.update(ids)
        result[split] = arrays
    return manifest, result


def encode_ids(values):
    """Preserve numeric identifiers; hash full textual IDs, never strip characters."""
    reverse = {}
    encoded = {}
    for value in values:
        value = str(value)
        number = int(value) if value.isascii() and value.isdecimal() else -1
        if not 0 <= number < 2**63:
            number = int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big") & (
                2**63 - 1
            )
        if number in reverse and reverse[number] != value:
            raise ValueError(f"ID encoding collision: {value!r} and {reverse[number]!r}")
        reverse[number] = value
        encoded[value] = number
    return encoded
