"""Training loop. Called directly by run_pipeline.py (no job queue)."""
import torch


class Trainer:
    def __init__(self, model, train_cfg: dict, model_cfg: dict, run_dir: str):
        # TODO: optimizer, loss, device, early-stopping state
        raise NotImplementedError

    def train_epoch(self, loader) -> float:
        # TODO: forward, loss, backward, grad clip, step
        raise NotImplementedError

    @torch.no_grad()
    def evaluate(self, loader):
        """Return (loss, y_true, y_pred) in scaled units."""
        # TODO
        raise NotImplementedError

    def fit(self, train_loader, val_loader) -> dict:
        """Run epochs, save best checkpoint, return history {'train_loss': [...], 'val_loss': [...]}."""
        # TODO
        raise NotImplementedError
