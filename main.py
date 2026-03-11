"""SSMD CLI entry point."""

import argparse
from ssmd.train import train


def main():
    parser = argparse.ArgumentParser(
        description="Train SSMD (Semi-Supervised Medical Image Detector)"
    )
    parser.add_argument(
        "--dataset",
        choices=["dsb", "deeplesion"],
        default="dsb",
        help="Dataset to train on (default: dsb)",
    )
    parser.add_argument(
        "--labeled-fraction",
        type=float,
        default=0.2,
        help="Fraction of training images used as labeled (default: 0.2)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size (default: 8)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to dataset root (default: dataset/<dataset-name>)",
    )
    args = parser.parse_args()
    config = {
        "dataset": args.dataset,
        "labeled_fraction": args.labeled_fraction,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
    }
    if args.data_dir:
        config["data_dir"] = args.data_dir

    train(config)


if __name__ == "__main__":
    main()
