import argparse


DATASETS = ("icip", "smpd", "instagram")


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
    parser.add_argument("--data-dir", default="data", help="Directory containing icip/smpd/instagram split pickles.")
    parser.add_argument("--config", default="src/config/config.yaml", help="YAML configuration file.")
    parser.add_argument("--output-dir", dest="save_path", default="results")

    parser.add_argument("--batch-size", dest="batch_size", type=int, default=512)
    parser.add_argument("--embedding-size", dest="embSize", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=None, help="Override retrieval.top_k from config.")
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--num-workers", type=int, default=0)

    if require_model_path:
        parser.add_argument("--model-path", required=True)
        parser.add_argument("--tta-runs", type=int, default=None)
        parser.add_argument(
            "--tta-noise-stds",
            default=None,
            help="Comma-separated multi-scale TTA noise levels, e.g. 0.003,0.006,0.01.",
        )
        parser.add_argument("--tta-confidence-temperature", type=float, default=None)
        parser.add_argument("--tta-anchor-weight", type=float, default=None)
        parser.add_argument("--no-tta", action="store_true", help="Disable test-time perturbation.")
    else:
        parser.add_argument("--epochs", dest="epoch", type=int, default=100)
        parser.add_argument("--lr", type=float, default=None, help="Override training.learning_rate from config.")
        parser.add_argument("--weight-decay", type=float, default=None)
        parser.add_argument("--ib-loss-weight", type=float, default=None)
        parser.add_argument(
            "--run-name",
            default=None,
            help="Optional experiment name. Defaults to a timestamped run directory.",
        )
        parser.add_argument("--resume-path", default=None)
        parser.add_argument("--patience", type=int, default=10)

    return parser
