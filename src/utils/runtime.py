import copy
import logging
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

from utils.data_loader import DEFAULT_META_FIELDS, get_all_items


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def meta_fields_for(config: Dict[str, Any], data_name: str) -> Sequence[str]:
    configured = config.get("data", {}).get("meta_fields", {})
    if data_name in configured:
        return configured[data_name]
    return DEFAULT_META_FIELDS.get(data_name, [])


def prepare_config(config: Dict[str, Any], args, meta_dim: int) -> Dict[str, Any]:
    cfg = copy.deepcopy(config)
    cfg.setdefault("retrieval", {})
    cfg.setdefault("training", {})

    if args.top_k is not None:
        cfg["retrieval"]["top_k"] = args.top_k
    cfg["retrieval"].setdefault("top_k", 50)
    cfg["retrieval"].setdefault("mol", {})
    cfg["retrieval"]["mol"]["query_embedding_dim"] = args.embSize * 2 + meta_dim
    cfg["retrieval"]["mol"]["item_embedding_dim"] = args.embSize * 2 + meta_dim

    if getattr(args, "lr", None) is not None:
        cfg["training"]["learning_rate"] = args.lr
    if getattr(args, "weight_decay", None) is not None:
        cfg["training"]["weight_decay"] = args.weight_decay
    if getattr(args, "ib_loss_weight", None) is not None:
        cfg["training"]["ib_loss_weight"] = args.ib_loss_weight
    cfg["training"].setdefault("learning_rate", 1e-4)
    cfg["training"].setdefault("weight_decay", 0.0)
    cfg["training"].setdefault("ib_loss_weight", 1.0)
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(name)


def setup_logging(output_dir: Path, data_name: Optional[str] = None, log_name: str = "train.log") -> logging.Logger:
    log_dir = output_dir / data_name / "logs" if data_name is not None else output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("jrpp")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_dir / log_name, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def move_items_to_device(data: Iterable, device: torch.device) -> Tuple[torch.Tensor, ...]:
    return tuple(tensor.to(device) for tensor in get_all_items(list(data)))


def build_optimizer(model: torch.nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    training = config.get("training", {})
    return torch.optim.Adam(
        model.parameters(),
        lr=float(training.get("learning_rate", 1e-4)),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
