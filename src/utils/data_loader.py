import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


DEFAULT_META_FIELDS: Dict[str, List[str]] = {
    "icip": ["mean_views"],
    "smpd": [],
    "instagram": [],
}

SPLIT_FILES = {
    "train": "train.pkl",
    "val": "val.pkl",
    "test": "test.pkl",
}


@dataclass(frozen=True)
class PopularitySample:
    image_id: torch.Tensor
    user_id: torch.Tensor
    text_vec: torch.Tensor
    img_vec_cls: torch.Tensor
    meta_features: torch.Tensor
    y: torch.Tensor
    img_vec_pool: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class PopularityBatch:
    image_id: torch.Tensor
    user_id: torch.Tensor
    text_vec: torch.Tensor
    img_vec_cls: torch.Tensor
    meta_features: torch.Tensor
    y: torch.Tensor
    img_vec_pool: Optional[torch.Tensor] = None

    def to(self, device: torch.device):
        return PopularityBatch(
            image_id=self.image_id.to(device),
            user_id=self.user_id.to(device),
            text_vec=self.text_vec.to(device),
            img_vec_cls=self.img_vec_cls.to(device),
            meta_features=self.meta_features.to(device),
            y=self.y.to(device),
            img_vec_pool=self.img_vec_pool.to(device) if self.img_vec_pool is not None else None,
        )


def _dataset_dir(data_name: str, data_dir: str) -> Path:
    return Path(data_dir) / data_name


def _split_path(data_name: str, data_dir: str, split: str) -> Path:
    if split not in SPLIT_FILES:
        raise ValueError(f"Unknown split {split!r}. Expected one of {sorted(SPLIT_FILES)}.")
    return _dataset_dir(data_name, data_dir) / SPLIT_FILES[split]


def _as_scalar(value, default=0):
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            return default
        return value[0]
    return value


def _to_int_id(value) -> int:
    value = _as_scalar(value, default=0)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return int(value)

    digits = re.sub(r"\D", "", str(value))
    return int(digits) if digits else 0


def _feature_tensor(value, field_name: str) -> torch.Tensor:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        raise ValueError(f"{field_name} must be a vector, got a scalar.")
    return torch.as_tensor(array.reshape(-1), dtype=torch.float32)


def _label_tensor(value) -> torch.Tensor:
    return torch.tensor([float(_as_scalar(value, default=0.0))], dtype=torch.float32)


def _numeric_values(value) -> List[float]:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return [float(value)]
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(item) for item in value if isinstance(item, (int, float, np.integer, np.floating))]
    return []


def _row_to_sample(row, meta_fields: Sequence[str]) -> PopularitySample:
    if "merged_text_vec" in row:
        text_vec = _feature_tensor(row["merged_text_vec"], "merged_text_vec")
    elif "text_vec" in row:
        text_vec = _feature_tensor(row["text_vec"], "text_vec")
    else:
        raise KeyError("Expected a text embedding field: merged_text_vec or text_vec.")

    if "cls_vec" in row:
        img_vec_cls = _feature_tensor(row["cls_vec"], "cls_vec")
    elif "img_vec_cls" in row:
        img_vec_cls = _feature_tensor(row["img_vec_cls"], "img_vec_cls")
    else:
        raise KeyError("Expected an image embedding field: cls_vec or img_vec_cls.")

    img_vec_pool = None
    if "mean_pooling_vec" in row:
        img_vec_pool = _feature_tensor(row["mean_pooling_vec"], "mean_pooling_vec")

    return PopularitySample(
        image_id=torch.tensor(_to_int_id(row.get("image_id", 0)), dtype=torch.long),
        user_id=torch.tensor(_to_int_id(row.get("user_id", 0)), dtype=torch.long),
        text_vec=text_vec,
        img_vec_cls=img_vec_cls,
        meta_features=torch.tensor(
            [value for key in meta_fields for value in _numeric_values(row.get(key, []))],
            dtype=torch.float32,
        ),
        y=_label_tensor(row.get("label", row.get("label_log2", 0.0))),
        img_vec_pool=img_vec_pool,
    )


def _read_pickle_split(path: Path, meta_fields: Sequence[str]) -> List[PopularitySample]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset split not found: {path}")

    with path.open("rb") as f:
        data = pickle.load(f)

    if isinstance(data, list):
        data = pd.DataFrame(data)
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame or list in {path}, got {type(data)!r}.")

    return [_row_to_sample(row, meta_fields) for _, row in data.iterrows()]


def read_data(
    data_name: str,
    data_dir: str = "data",
    meta_fields: Optional[Sequence[str]] = None,
    splits: Iterable[str] = ("train", "val", "test"),
):
    fields = DEFAULT_META_FIELDS.get(data_name, []) if meta_fields is None else list(meta_fields)
    return tuple(
        _read_pickle_split(_split_path(data_name, data_dir, split), meta_fields=fields)
        for split in splits
    )


def get_all_items(
    data: Sequence[PopularitySample],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    item_ids = torch.stack([item.image_id for item in data], dim=0)
    item_text = torch.stack([item.text_vec for item in data], dim=0)
    item_image = torch.stack([item.img_vec_cls for item in data], dim=0)
    item_meta = torch.stack([item.meta_features for item in data], dim=0).unsqueeze(1)
    return item_ids, item_text, item_image, item_meta


def collate_popularity_batch(samples: Sequence[PopularitySample]) -> PopularityBatch:
    img_vec_pool = None
    if all(sample.img_vec_pool is not None for sample in samples):
        img_vec_pool = torch.stack([sample.img_vec_pool for sample in samples], dim=0)

    return PopularityBatch(
        image_id=torch.stack([sample.image_id for sample in samples], dim=0),
        user_id=torch.stack([sample.user_id for sample in samples], dim=0),
        text_vec=torch.stack([sample.text_vec for sample in samples], dim=0),
        img_vec_cls=torch.stack([sample.img_vec_cls for sample in samples], dim=0),
        meta_features=torch.stack([sample.meta_features for sample in samples], dim=0),
        y=torch.stack([sample.y for sample in samples], dim=0),
        img_vec_pool=img_vec_pool,
    )


class PopularityDataset(Dataset):
    def __init__(self, data: Sequence[PopularitySample]):
        self.data = list(data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> PopularitySample:
        return self.data[index]
