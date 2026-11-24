import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from networks.dual_branch_net import DualBranchNet
from training.checkpoint import load_model, save_checkpoint
from training.trainer import Trainer


def _toy(cfg, n=256):
    d = cfg["derived"]
    g = torch.Generator().manual_seed(0)
    spatial = torch.randn(n, d["C"], 2, 2, generator=g)
    temporal = torch.randn(n, d["T"], d["F"], generator=g)
    y = (0.8 * temporal[:, -1, 0] + 0.3 * spatial[:, 0].mean(dim=(1, 2))).unsqueeze(1)
    return TensorDataset(spatial, temporal, y)


def test_loss_drops_on_a_tiny_subset(cfg, tmp_path):
    tc = {**cfg["train"], "epochs": 40, "batch_size": 32, "device": "cpu", "early_stopping_patience": 100}
    torch.manual_seed(0)
    trainer = Trainer(DualBranchNet(cfg["model"]), tc, cfg["model"], str(tmp_path))
    loader = DataLoader(_toy(cfg), batch_size=32, shuffle=True)
    hist = trainer.fit(loader, None, verbose=False)
    assert hist["train_loss"][-1] < 0.5 * hist["train_loss"][0]


def test_early_stopping_restores_best_epoch(cfg, tmp_path):
    tc = {**cfg["train"], "epochs": 30, "device": "cpu", "early_stopping_patience": 2}
    trainer = Trainer(DualBranchNet(cfg["model"]), tc, cfg["model"], str(tmp_path))
    loader = DataLoader(_toy(cfg), batch_size=64)
    hist = trainer.fit(loader, loader, verbose=False)
    assert hist["best_epoch"] is not None and hist["best_epoch"] <= hist["epochs_run"]


def test_checkpoint_roundtrip_rebuilds_the_same_model(cfg, tmp_path):
    model = DualBranchNet(cfg["model"]).eval()
    path = str(tmp_path / "m.pt")
    save_checkpoint(path, model, cfg["model"], cfg["train"], epoch=3, val_loss=0.5)
    loaded, ckpt = load_model(path)
    d = cfg["derived"]
    x = (torch.randn(3, d["C"], 2, 2), torch.randn(3, d["T"], d["F"]))
    assert torch.allclose(model(*x), loaded(*x))
    assert ckpt["model_cfg"] == cfg["model"] and ckpt["epoch"] == 3
