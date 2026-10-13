import argparse
import math

DATASETS = ("icip", "smpd", "instagram")


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return value


def seed_value(value):
    value = nonnegative_int(value)
    if value >= 2**32:
        raise argparse.ArgumentTypeError("seed must be below 2**32")
    return value


def nonnegative_float(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return value


def positive_float(value):
    value = nonnegative_float(value)
    if value == 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def dropout_rate(value):
    value = nonnegative_float(value)
    if value >= 1:
        raise argparse.ArgumentTypeError("dropout must be in [0, 1)")
    return value


def normalize_dataset_name(value: str) -> str:
    return value.lower()


def build_parser(require_model_path: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate JRPP for multimodal social media popularity prediction."
            if require_model_path
            else "Train JRPP for multimodal social media popularity prediction."
        )
    )

    parser.add_argument(
        "--data-name",
        dest="data_name",
        type=normalize_dataset_name,
        choices=DATASETS,
        default="icip",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing prepared icip/smpd/instagram datasets.",
    )
    parser.add_argument(
        "--config", default="src/config/config.yaml", help="YAML configuration file."
    )
    parser.add_argument("--output-dir", dest="save_path", default="results")

    parser.add_argument("--batch-size", dest="batch_size", type=positive_int, default=512)
    parser.add_argument("--embedding-size", dest="embSize", type=positive_int, default=512)
    parser.add_argument("--dropout", type=dropout_rate, default=0.2)
    parser.add_argument(
        "--top-k", type=positive_int, default=None, help="Override retrieval.top_k from config."
    )
    parser.add_argument("--seed", type=seed_value, default=12)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--num-workers", type=nonnegative_int, default=0)

    if require_model_path:
        parser.set_defaults(data_name=None, embSize=None, dropout=None, config=None)
        parser.add_argument("--model-path", required=True)
        parser.add_argument(
            "--tta-runs",
            type=int,
            default=None,
            help="Even number of paired perturbations (default: 4); 0 disables TTA.",
        )
        parser.add_argument(
            "--tta-relative-scale",
            type=float,
            default=0.02,
            help="Noise L2 norm relative to each modality's feature norm (default: 0.02).",
        )
        parser.add_argument(
            "--tta-anchor-weight",
            type=float,
            default=0.5,
            help="Original prediction's weight in the average (default: 0.5).",
        )
        parser.add_argument(
            "--no-tta", action="store_true", help="Use plain prediction without perturbation."
        )
        parser.add_argument(
            "--no-retrieval-cache",
            action="store_true",
            help="Disable the evaluation-only retrieval bank cache.",
        )
    else:
        parser.add_argument("--epochs", dest="epoch", type=positive_int, default=100)
        parser.add_argument(
            "--lr",
            type=positive_float,
            default=None,
            help="Override training.learning_rate from config.",
        )
        parser.add_argument("--weight-decay", type=nonnegative_float, default=None)
        parser.add_argument("--ib-loss-weight", type=nonnegative_float, default=None)
        parser.add_argument(
            "--run-name",
            default=None,
            help="Optional experiment name. Defaults to a timestamped run directory.",
        )
        parser.add_argument("--resume-path", default=None)
        parser.add_argument(
            "--keep-epoch-checkpoints",
            action="store_true",
            help="Also retain a full resume checkpoint for each epoch.",
        )
        parser.add_argument("--patience", type=nonnegative_int, default=10)

    return parser
