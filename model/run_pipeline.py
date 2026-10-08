"""CLI entry point.

    python run_pipeline.py --stage preprocess
    python run_pipeline.py --stage train
    python run_pipeline.py --stage report [--run-id 20260101_120000]
"""
import argparse
from datetime import datetime


def preprocess(cfg: dict) -> None:
    """clean_loader -> windows -> patches -> align -> split -> fit scalers on train -> save 9 .npy + scalers."""
    # TODO
    raise NotImplementedError


def train(cfg: dict) -> str:
    """set_seed, build loaders and model, Trainer.fit, save best_dual_branch_model.pt.
    Returns run_id (timestamp, e.g. datetime.now().strftime("%Y%m%d_%H%M%S"))."""
    # TODO
    raise NotImplementedError


def report(cfg: dict, run_id: str | None) -> None:
    """generate_snapshot for run_id (default: latest run)."""
    # TODO
    raise NotImplementedError


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=["preprocess", "train", "report"])
    p.add_argument("--run-id", default=None)
    args = p.parse_args()

    # TODO: cfg = load_config(); set_seed(cfg["train"]["seed"])
    cfg = {}
    if args.stage == "preprocess":
        preprocess(cfg)
    elif args.stage == "train":
        train(cfg)
    else:
        report(cfg, args.run_id)


if __name__ == "__main__":
    main()
