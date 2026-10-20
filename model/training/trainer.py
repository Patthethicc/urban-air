"""Training loop. Called directly by run_pipeline.py (no job queue)."""
import copy

import numpy as np
import torch
from torch.utils.data import DataLoader

from runtime_guard import Progress, guarded
from training.checkpoint import save_checkpoint
from training.loss import build_loss


def resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def make_loader(dataset, train_cfg: dict, shuffle: bool, batch_size: int | None = None) -> DataLoader:
    gen = torch.Generator().manual_seed(int(train_cfg["seed"]))
    return DataLoader(dataset, batch_size=batch_size or train_cfg["batch_size"], shuffle=shuffle,
                      num_workers=train_cfg["num_workers"], generator=gen if shuffle else None)


class Trainer:
    def __init__(self, model, train_cfg: dict, model_cfg: dict, run_dir: str, extra: dict | None = None):
        self.train_cfg, self.model_cfg, self.run_dir, self.extra = train_cfg, model_cfg, run_dir, extra or {}
        self.device = resolve_device(train_cfg["device"])
        self.model = model.to(self.device)
        self.model.dedupe_patches = True               # station samples share patches: see SpatialCNN.forward
        o = train_cfg["optimizer"]
        opt = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[o["name"]]
        self.optimizer = opt(self.model.parameters(), lr=o["lr"], weight_decay=o["weight_decay"])
        self.criterion = build_loss(train_cfg["loss"])

    def train_epoch(self, loader, label: str = "training", verbose: bool = True) -> float:
        """One pass. `label` names it in progress lines and in a timeout message (see runtime_guard.py)."""
        self.model.train()
        total, n = 0.0, 0
        progress = Progress(len(loader), label, enabled=verbose, announce=True)
        with guarded(label, quiet=not verbose) as guard:
            for i, (spatial, temporal, target) in enumerate(loader, 1):
                spatial, temporal, target = spatial.to(self.device), temporal.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(spatial, temporal), target)
                loss.backward()
                if self.train_cfg["grad_clip"]:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_cfg["grad_clip"])
                self.optimizer.step()
                total += loss.item() * len(target)
                n += len(target)
                guard.beat(f"{label}, batch {i}/{len(loader)}")
                progress.update(i)
        return total / max(n, 1)

    @torch.no_grad()
    def evaluate(self, loader, label: str = "evaluation", verbose: bool = False):
        """Return (loss, y_true, y_pred) in scaled units, arrays of shape [N, 1]."""
        self.model.eval()
        total, n, ys, ps = 0.0, 0, [], []
        progress = Progress(len(loader), label, enabled=verbose)
        with guarded(label, quiet=not verbose) as guard:
            for i, (spatial, temporal, target) in enumerate(loader, 1):
                spatial, temporal, target = spatial.to(self.device), temporal.to(self.device), target.to(self.device)
                pred = self.model(spatial, temporal)
                total += self.criterion(pred, target).item() * len(target)
                n += len(target)
                ys.append(target.cpu().numpy())
                ps.append(pred.cpu().numpy())
                guard.beat(f"{label}, batch {i}/{len(loader)}")
                progress.update(i)
        return total / max(n, 1), np.concatenate(ys), np.concatenate(ps)

    def fit(self, train_loader, val_loader=None, epochs: int | None = None, ckpt_path: str | None = None,
            verbose: bool = True, tag: str = "") -> dict:
        """Run epochs and return a history dict.

        With a val_loader and train.early_stopping: stop on val loss, best weights restored (history['best_epoch']).
        Without a val_loader, or with early_stopping false: exactly `epochs` epochs, final weights kept."""
        epochs = int(epochs or self.train_cfg["epochs"])
        patience = int(self.train_cfg["early_stopping_patience"])
        stopping = bool(self.train_cfg.get("early_stopping", True))
        hist = {"train_loss": [], "val_loss": [], "best_epoch": None, "best_val_loss": None, "epochs_run": 0}
        best_state, best_val, since_best = None, float("inf"), 0

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader, f"{tag} epoch {epoch}/{epochs}".strip(), verbose)
            hist["train_loss"].append(train_loss)
            msg = f"{tag} epoch {epoch:3d}/{epochs}  train {train_loss:.4f}"
            if val_loader is not None:
                val_loss, _, _ = self.evaluate(val_loader, f"{tag} validation (epoch {epoch})".strip(), verbose)
                hist["val_loss"].append(val_loss)
                msg += f"  val {val_loss:.4f}"
                if val_loss < best_val - 1e-8:
                    best_val, since_best = val_loss, 0
                    best_state = copy.deepcopy(self.model.state_dict())
                    hist["best_epoch"], hist["best_val_loss"] = epoch, val_loss
                else:
                    since_best += 1
            hist["epochs_run"] = epoch
            if verbose:
                print(msg)
            if stopping and val_loader is not None and since_best >= patience:
                if verbose:
                    print(f"{tag} early stop (no val improvement for {patience} epochs)")
                break

        if best_state is not None and stopping:          # keep the final weights when `epochs` is fixed
            self.model.load_state_dict(best_state)
        if hist["best_epoch"] is None:
            hist["best_epoch"] = hist["epochs_run"]
        if ckpt_path:
            save_checkpoint(ckpt_path, self.model, self.model_cfg, self.train_cfg,
                            hist["best_epoch"], hist["best_val_loss"], self.extra)
        return hist
