import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.JRPP import JRPP
from models.initialization import initialize_output_bias_from_train
from utils.data_loader import PopularityDataset, collate_popularity_batch, read_data
from utils.metrics import regression_metrics
from utils.parsers import build_parser
from utils.user_identity import checkpoint_identity, configure_training_identity, initialize_user_identity
from utils.runtime import (
    atomic_torch_save,
    build_optimizer,
    capture_random_state,
    load_config,
    meta_fields_for,
    move_items_to_device,
    prepare_config,
    resolve_device,
    restore_random_state,
    seed_everything,
    setup_logging,
)

FRESH_RNG_POLICY = "post-init-seed-v1"


def _batch_inputs(batch, items: Tuple[torch.Tensor, ...]):
    items_id, items_text, items_img, items_meta = items
    return (
        batch.text_vec,
        batch.meta_features,
        batch.img_vec_cls,
        batch.user_id,
        batch.image_id,
        items_id,
        items_text,
        items_img,
        items_meta,
    )


def train_one_epoch(
    model: JRPP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    items: Tuple[torch.Tensor, ...],
    device: torch.device,
    ib_loss_weight: float,
    epoch: int,
) -> Dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "mse_loss": 0.0, "kl_loss": 0.0, "count": 0}

    progress = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
    for batch in progress:
        batch = batch.to(device)
        target = batch.y.float()

        optimizer.zero_grad(set_to_none=True)
        pred, kl_loss = model(*_batch_inputs(batch, items))
        target = target.view_as(pred)
        mse_loss = criterion(pred, target)
        loss = mse_loss + ib_loss_weight * kl_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss; checkpoint was not updated.")
        loss.backward()
        optimizer.step()

        batch_size = int(target.numel())
        totals["loss"] += float(loss.detach().cpu()) * batch_size
        totals["mse_loss"] += float(mse_loss.detach().cpu()) * batch_size
        totals["kl_loss"] += float(kl_loss.detach().cpu()) * batch_size
        totals["count"] += batch_size
        progress.set_postfix(loss=totals["loss"] / max(1, totals["count"]))

    count = max(1, totals["count"])
    return {key: value / count for key, value in totals.items() if key != "count"}


@torch.no_grad()
def evaluate(
    model: JRPP,
    loader: DataLoader,
    items: Tuple[torch.Tensor, ...],
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    predictions = []
    targets = []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        batch = batch.to(device)
        pred, _ = model(*_batch_inputs(batch, items))
        predictions.append(pred.detach().cpu().reshape(-1).numpy())
        targets.append(batch.y.detach().cpu().reshape(-1).numpy())

    return regression_metrics(np.concatenate(predictions), np.concatenate(targets))


def save_checkpoint(
    path: Path,
    model: JRPP,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    args,
    config: Dict,
    metrics: Dict[str, float],
    bad_epochs: int = 0,
    resumable: bool = True,
) -> None:
    user_identity = checkpoint_identity({"config": config, "user_identity": getattr(model, "user_identity", None)})
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "best_metric": best_metric,
        "args": vars(args),
        "config": config,
        "metrics": metrics,
        "evaluation_protocol": "plain",
        "initialization_policy": model.initialization_policy,
        "fresh_rng_policy": model.fresh_rng_policy,
        "resume_source": getattr(model, "resume_source", None),
        "output_bias_initialization": getattr(model, "output_bias_initialization", None),
        "user_identity": user_identity,
        "user_embedding_initialization": getattr(model, "user_embedding_initialization", None),
    }
    if resumable:
        payload.update(
            optimizer_state_dict=optimizer.state_dict(),
            random_state=capture_random_state(),
            bad_epochs=bad_epochs,
            resume_format=1,
        )
    atomic_torch_save(payload, path)


def load_checkpoint(
    path: str, model: JRPP, optimizer: torch.optim.Optimizer, device: torch.device, args=None
):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint_identity(checkpoint) != model.config.get("user_identity"):
        raise ValueError("Checkpoint user identity differs from the current model configuration.")
    if checkpoint.get("config", {}).get("data_protocol") != model.config.get("data_protocol"):
        raise ValueError(
            "Checkpoint data protocol differs; prepare matching data or start a new run."
        )
    required_state = ("epoch", "best_metric", "bad_epochs", "optimizer_state_dict", "random_state",
                      "initialization_policy", "fresh_rng_policy")
    if checkpoint.get("resume_format") != 1 or any(key not in checkpoint for key in required_state):
        raise ValueError("Checkpoint has no complete resume state; use checkpoints/JRPP_last.pt.")
    if checkpoint["config"] != model.config:
        raise ValueError("Resume configuration differs from the saved training configuration.")
    if args is not None:
        for key in (
            "data_name",
            "embSize",
            "dropout",
            "batch_size",
            "seed",
            "num_workers",
            "patience",
        ):
            if checkpoint["args"].get(key) != getattr(args, key):
                raise ValueError(f"Resume training argument differs: {key}")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    restore_random_state(checkpoint["random_state"])
    model.initialization_policy = checkpoint["initialization_policy"]
    model.fresh_rng_policy = checkpoint["fresh_rng_policy"]
    model.output_bias_initialization = checkpoint.get("output_bias_initialization")
    model.user_embedding_initialization = checkpoint.get("user_embedding_initialization")
    with Path(path).open("rb") as stream:
        source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    model.resume_source = {"path": str(Path(path).resolve()), "sha256": source_hash}
    return checkpoint["epoch"] + 1, checkpoint["best_metric"], checkpoint["bad_epochs"]


def _safe_run_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    if not name or name in (".", ".."):
        raise ValueError("Run name cannot be empty.")
    return name


def _unique_run_dir(base_dir: Path) -> Path:
    if not base_dir.exists():
        return base_dir
    for index in range(2, 1000):
        candidate = base_dir.with_name(f"{base_dir.name}_{index:02d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Unable to create a unique run directory under {base_dir.parent}.")


def _resume_run_dir(resume_path: str) -> Path | None:
    path = Path(resume_path)
    if path.parent.name == "checkpoints":
        return path.parent.parent
    return None


def resolve_run_dir(
    output_dir: Path, data_name: str, run_name: str | None, resume_path: str | None
) -> Path:
    if resume_path and run_name is None:
        resume_dir = _resume_run_dir(resume_path)
        if resume_dir is not None:
            return resume_dir

    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _unique_run_dir(output_dir / data_name / _safe_run_name(run_name))


def main() -> None:
    args = build_parser().parse_args()
    seed_everything(args.seed)

    raw_config = load_config(args.config)
    meta_fields = meta_fields_for(raw_config, args.data_name)
    train_split, val_split = read_data(
        args.data_name,
        data_dir=args.data_dir,
        meta_fields=meta_fields,
        splits=("train", "val"),
    )

    if len(train_split) == 0 or len(val_split) == 0:
        raise RuntimeError("Training and validation splits must both be non-empty.")

    resume_identity = torch.load(args.resume_path, map_location="cpu", weights_only=False) if args.resume_path else None
    user_identity = configure_training_identity(train_split, val_split,
        checkpoint=resume_identity)
    del resume_identity

    meta_dim = int(train_split[0].meta_features.size(-1))
    config = prepare_config(raw_config, args, meta_dim)
    config["data_protocol"] = train_split.protocol
    config["user_identity"] = user_identity
    device = resolve_device(args.device)

    run_dir = resolve_run_dir(Path(args.save_path), args.data_name, args.run_name, args.resume_path)
    logger = setup_logging(run_dir, log_name="train.log")
    logger.info(
        "Dataset=%s train=%d val=%d meta_dim=%d device=%s seed=%d run_dir=%s",
        args.data_name,
        len(train_split),
        len(val_split),
        meta_dim,
        device,
        args.seed,
        run_dir,
    )
    logger.info("Meta fields: %s", list(meta_fields))

    train_loader = DataLoader(
        PopularityDataset(train_split),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_popularity_batch,
    )
    val_loader = DataLoader(
        PopularityDataset(val_split),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_popularity_batch,
    )

    train_items = move_items_to_device(train_split, device)
    val_items = train_items

    model = JRPP(args=args, config=config, meta_dim=meta_dim, dropout=args.dropout).to(device)
    optimizer = build_optimizer(model, config)
    criterion = nn.MSELoss()

    start_epoch = 0
    best_mse = float("inf")
    bad_epochs = 0
    if args.resume_path:
        start_epoch, best_mse, bad_epochs = load_checkpoint(
            args.resume_path, model, optimizer, device, args
        )
        logger.info(
            "Resumed from %s at epoch %d with best validation MSE %.6f",
            args.resume_path,
            start_epoch,
            best_mse,
        )
    else:
        initialize_output_bias_from_train(model, train_split, config["data_protocol"])
        initialize_user_identity(model)
        logger.info("Training authors=%d; vocabulary SHA=%s",
            len(user_identity["vocabulary"]), user_identity["vocabulary_sha256"])
        # Start training randomness independently of parameter initialization,
        # while resumed runs continue from the saved random state.
        seed_everything(args.seed)
        model.fresh_rng_policy = FRESH_RNG_POLICY
        model.resume_source = None
        logger.info("Fresh output-bias initialization: %s", model.output_bias_initialization)
        logger.info("Fresh training RNG reset after model initialization: seed=%d", args.seed)

    logger.info("Initialization policy=%s; fresh RNG policy=%s",
                model.initialization_policy, model.fresh_rng_policy)

    checkpoint_dir = run_dir / "checkpoints"
    best_model_path = run_dir / "JRPP_best.pt"
    ib_loss_weight = float(config.get("training", {}).get("ib_loss_weight", 1.0))

    for epoch in range(start_epoch, args.epoch):
        if args.patience > 0 and bad_epochs >= args.patience:
            logger.info("Saved run already reached early stopping; no further epochs executed.")
            break
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            train_items,
            device,
            ib_loss_weight,
            epoch,
        )
        val_metrics = evaluate(model, val_loader, val_items, device)

        improved = val_metrics["mse"] < best_mse
        best_marker = " *best*" if improved else ""
        logger.info(
            "epoch=%d train_loss=%.6f train_mse=%.6f val_mse=%.6f val_mae=%.6f val_src=%.6f%s",
            epoch,
            train_metrics["loss"],
            train_metrics["mse_loss"],
            val_metrics["mse"],
            val_metrics["mae"],
            val_metrics["src"],
            best_marker,
        )

        if improved:
            best_mse = val_metrics["mse"]
            bad_epochs = 0
            save_checkpoint(
                best_model_path,
                model,
                optimizer,
                epoch,
                best_mse,
                args,
                config,
                val_metrics,
                bad_epochs=bad_epochs,
                resumable=False,
            )
        else:
            bad_epochs += 1

        save_checkpoint(
            checkpoint_dir / "JRPP_last.pt",
            model,
            optimizer,
            epoch,
            best_mse,
            args,
            config,
            val_metrics,
            bad_epochs=bad_epochs,
        )
        if args.keep_epoch_checkpoints:
            save_checkpoint(
                checkpoint_dir / f"JRPP_epoch_{epoch}.pt",
                model,
                optimizer,
                epoch,
                best_mse,
                args,
                config,
                val_metrics,
                bad_epochs=bad_epochs,
            )

        if args.patience > 0 and bad_epochs >= args.patience:
            logger.info(
                "Early stopping after %d epochs without validation improvement.", bad_epochs
            )
            break

    logger.info("Finished training. Best validation MSE: %.6f", best_mse)


if __name__ == "__main__":
    main()
