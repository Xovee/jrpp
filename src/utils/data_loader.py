from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

DEFAULT_META_FIELDS: Dict[str, List[str]] = {
    "icip": ["mean_views"],
    "smpd": [],
    "instagram": [],
}


@dataclass(frozen=True)
class PopularitySample:
    image_id: torch.Tensor
    user_id: torch.Tensor
    text_vec: torch.Tensor
    img_vec_cls: torch.Tensor
    meta_features: torch.Tensor
    y: torch.Tensor


@dataclass(frozen=True)
class PopularityBatch:
    image_id: torch.Tensor
    user_id: torch.Tensor
    text_vec: torch.Tensor
    img_vec_cls: torch.Tensor
    meta_features: torch.Tensor
    y: torch.Tensor

    def to(self, device: torch.device):
        return PopularityBatch(
            image_id=self.image_id.to(device),
            user_id=self.user_id.to(device),
            text_vec=self.text_vec.to(device),
            img_vec_cls=self.img_vec_cls.to(device),
            meta_features=self.meta_features.to(device),
            y=self.y.to(device),
        )


def _dataset_dir(data_name: str, data_dir: str) -> Path:
    return Path(data_dir) / data_name


class PreparedSplit(Sequence):
    def __init__(self, arrays, image_codes, fields, protocol):
        self.original_ids = arrays["image_id"].tolist()
        self.original_user_ids = arrays["user_id"].tolist()
        self.protocol = protocol
        self.image_ids = torch.tensor([image_codes[x] for x in self.original_ids], dtype=torch.long)
        self.user_ids = None
        self.text = torch.from_numpy(arrays["merged_text_vec"])
        self.image = torch.from_numpy(arrays["cls_vec"])
        self.labels = torch.from_numpy(arrays["label"]).unsqueeze(1)
        self.meta = (
            torch.from_numpy(np.stack([arrays[x] for x in fields], axis=1))
            if fields
            else torch.empty(len(self.original_ids), 0)
        )

    def __len__(self):
        return len(self.original_ids)

    def __getitem__(self, index):
        if self.user_ids is None:
            raise ValueError('Configure the training-author vocabulary before using model inputs')
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        return PopularitySample(
            self.image_ids[index],
            self.user_ids[index],
            self.text[index],
            self.image[index],
            self.meta[index],
            self.labels[index],
        )


def read_data(data_name, data_dir="data", meta_fields=None, splits=("train", "val", "test")):
    from utils.data_contract import digest, encode_ids, load_contract

    source = _dataset_dir(data_name, data_dir)
    if not (source / "dataset.json").is_file():
        raise FileNotFoundError(
            f"Prepared dataset missing at {source}. Run src/prepare_data.py to validate and align published data first."
        )
    fields = DEFAULT_META_FIELDS.get(data_name, []) if meta_fields is None else list(meta_fields)
    requested = ["valid" if x == "val" else x for x in splits]
    if len(set(requested)) != len(requested) or any(
        x not in ("train", "valid", "test") for x in requested
    ):
        raise ValueError("Invalid or repeated split names")
    manifest, arrays = load_contract(source, data_name, fields)
    image_codes = encode_ids(x for a in arrays.values() for x in a["image_id"])
    protocol = {
        "manifest_sha256": digest(source / "dataset.json"),
        "reference_archive_sha256": manifest["reference_archive_sha256"],
        "format": manifest["format"],
        "dataset": data_name,
        "meta_fields": fields,
        "id_encoding": "full-string-sha256-int63-or-decimal-v1",
    }
    return tuple(
        PreparedSplit(arrays[x], image_codes, fields, protocol) for x in requested
    )


def get_all_items(
    data: Sequence[PopularitySample],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(data, PreparedSplit):
        return data.image_ids, data.text, data.image, data.meta.unsqueeze(1)
    item_ids = torch.stack([item.image_id for item in data], dim=0)
    item_text = torch.stack([item.text_vec for item in data], dim=0)
    item_image = torch.stack([item.img_vec_cls for item in data], dim=0)
    item_meta = torch.stack([item.meta_features for item in data], dim=0).unsqueeze(1)
    return item_ids, item_text, item_image, item_meta


def collate_popularity_batch(samples: Sequence[PopularitySample]) -> PopularityBatch:
    return PopularityBatch(
        image_id=torch.stack([sample.image_id for sample in samples], dim=0),
        user_id=torch.stack([sample.user_id for sample in samples], dim=0),
        text_vec=torch.stack([sample.text_vec for sample in samples], dim=0),
        img_vec_cls=torch.stack([sample.img_vec_cls for sample in samples], dim=0),
        meta_features=torch.stack([sample.meta_features for sample in samples], dim=0),
        y=torch.stack([sample.y for sample in samples], dim=0),
    )


class PopularityDataset(Dataset):
    def __init__(self, data: Sequence[PopularitySample]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> PopularitySample:
        return self.data[index]
