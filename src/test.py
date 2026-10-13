import json
from argparse import Namespace
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.JRPP import JRPP
from utils.data_loader import PopularityDataset, collate_popularity_batch, read_data
from utils.user_identity import checkpoint_identity, restore_identity_to_splits
from utils.metrics import regression_metrics
from utils.parsers import build_parser
from utils.runtime import (
    meta_fields_for,
    move_items_to_device,
    resolve_device,
    seed_everything,
    setup_logging,
)
from utils.tta import predict_tta, validate_tta_settings


def resolve_tta_settings(args):
    if args.no_tta and args.tta_runs not in (None, 0):
        raise ValueError("--no-tta conflicts with --tta-runs.")
    runs = 0 if args.no_tta else (4 if args.tta_runs is None else args.tta_runs)
    validate_tta_settings(runs, args.tta_relative_scale, args.tta_anchor_weight)
    return runs, args.tta_relative_scale, args.tta_anchor_weight


def restore_evaluation_settings(args, checkpoint):
    if getattr(args, "config", None) is not None:
        raise ValueError(
            "Evaluation uses the checkpoint config; omit --config. Use --tta-* for TTA overrides."
        )
    saved_args = checkpoint.get("args")
    config = checkpoint.get("config")
    if not isinstance(saved_args, dict) or not isinstance(config, dict):
        raise ValueError("Checkpoint must contain training args and config.")
    checkpoint_identity(checkpoint)
    restored = Namespace(**vars(args))
    for key in ("data_name", "embSize", "dropout", "top_k"):
        if key not in saved_args:
            raise ValueError(f"Checkpoint is missing training argument: {key}")
        requested = getattr(args, key)
        if requested is not None and requested != saved_args[key]:
            raise ValueError(f"Evaluation {key} differs from the checkpoint training setting.")
        setattr(restored, key, saved_args[key])
    return restored, deepcopy(config)


def main() -> None:
    args = build_parser(require_model_path=True).parse_args()
    checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=False)
    args, raw_config = restore_evaluation_settings(args, checkpoint)
    tta_runs, tta_relative_scale, tta_anchor_weight = resolve_tta_settings(args)
    seed_everything(args.seed)

    meta_fields = meta_fields_for(raw_config, args.data_name)
    train_split, test_split = read_data(
        args.data_name,
        data_dir=args.data_dir,
        meta_fields=meta_fields,
        splits=("train", "test"),
    )
    restore_identity_to_splits(checkpoint, train_split, test_split)

    meta_dim = int(test_split[0].meta_features.size(-1))
    config = raw_config
    if config.get("data_protocol") != train_split.protocol:
        raise ValueError("Checkpoint data protocol differs or is missing; prepare matching data.")
    device = resolve_device(args.device)
    logger = setup_logging(Path(args.save_path), args.data_name, log_name="test.log")

    model = JRPP(args=args, config=config, meta_dim=meta_dim, dropout=args.dropout).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    items = move_items_to_device(train_split, device)

    logger.info(
        "Evaluation=%s runs=%d relative_scale=%.6f anchor_weight=%.6f retrieval_cache=%s",
        "tta" if tta_runs else "plain",
        tta_runs,
        tta_relative_scale,
        tta_anchor_weight,
        not args.no_retrieval_cache,
    )

    test_loader = DataLoader(
        PopularityDataset(test_split),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_popularity_batch,
    )

    predictions = []
    targets = []
    cache_context = nullcontext() if args.no_retrieval_cache else model.evaluation_cache(items)
    with cache_context:
        for batch in tqdm(test_loader, desc=f"Testing {args.data_name}"):
            batch = batch.to(device)
            pred = predict_tta(model, batch, items, tta_runs, tta_relative_scale, tta_anchor_weight)
            predictions.append(pred.detach().cpu().numpy())
            targets.append(batch.y.detach().cpu().reshape(-1).numpy())

    metrics = regression_metrics(np.concatenate(predictions), np.concatenate(targets))
    output = Path(args.save_path) / args.data_name
    output.mkdir(parents=True, exist_ok=True)
    mode = "plain" if tta_runs == 0 else "tta"
    np.save(output / f"test_predictions_{mode}.npy", np.concatenate(predictions))
    np.savez(
        output / f"test_samples_{mode}.npz",
        image_id=np.asarray(test_split.original_ids),
        label=np.concatenate(targets),
        prediction=np.concatenate(predictions),
    )
    report = {
        "metrics": metrics,
        "evaluation_protocol": mode,
        "seed": args.seed,
        "checkpoint": str(Path(args.model_path).resolve()),
        "epoch": checkpoint.get("epoch"),
        "data_protocol": config["data_protocol"],
        "tta_runs": tta_runs,
        "tta_anchor_weight": tta_anchor_weight,
        "tta_relative_scale": tta_relative_scale,
        "retrieval_cache": not args.no_retrieval_cache,
    }
    (output / f"test_metrics_{mode}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    logger.info("Loaded checkpoint from %s", args.model_path)
    if checkpoint.get("epoch") is not None:
        logger.info("Checkpoint epoch: %s", checkpoint["epoch"])
    logger.info("Test MSE: %.6f", metrics["mse"])
    logger.info("Test MAE: %.6f", metrics["mae"])
    logger.info("Test SRC: %.6f", metrics["src"])


if __name__ == "__main__":
    main()
