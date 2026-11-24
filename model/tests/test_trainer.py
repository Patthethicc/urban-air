from pathlib import Path

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


def test_a_stopped_fit_continues_after_its_last_finished_epoch(cfg, tmp_path):
    """state_path: stop after epoch 2 of 5, start again, and the fit goes on at epoch 3 (not 1) with the same history and a finished state file removed by the caller."""
    tc = {**cfg["train"], "epochs": 5, "batch_size": 32, "device": "cpu", "early_stopping_patience": 100}
    loader = DataLoader(_toy(cfg), batch_size=32, shuffle=False)
    state = str(tmp_path / "fold_0_state.pt")

    class Stop(Exception):
        pass

    torch.manual_seed(0)
    first = Trainer(DualBranchNet(cfg["model"]), tc, cfg["model"], str(tmp_path))
    real = first.train_epoch
    calls = {"n": 0}

    def stopping_epoch(*a, **k):
        calls["n"] += 1
        if calls["n"] == 3:                        # the third epoch is the one running when the process dies
            raise Stop
        return real(*a, **k)

    first.train_epoch = stopping_epoch
    try:
        first.fit(loader, loader, verbose=False, state_path=state)
    except Stop:
        pass
    assert Path(state).is_file()                   # the state of the two finished epochs

    torch.manual_seed(1)
    second = Trainer(DualBranchNet(cfg["model"]), tc, cfg["model"], str(tmp_path))
    ran = []
    real2 = second.train_epoch
    second.train_epoch = lambda loader_, label, verbose: (ran.append(label), real2(loader_, label, verbose))[1]
    hist = second.fit(loader, loader, verbose=False, state_path=state)
    assert len(ran) == 3 and "epoch 3/5" in ran[0]          # epochs 3, 4, 5 only
    assert hist["epochs_run"] == 5 and len(hist["train_loss"]) == 5 and len(hist["val_loss"]) == 5


def test_a_state_made_with_other_settings_is_ignored(cfg, tmp_path):
    tc = {**cfg["train"], "epochs": 3, "batch_size": 32, "device": "cpu", "early_stopping_patience": 100}
    loader = DataLoader(_toy(cfg), batch_size=32)
    state = str(tmp_path / "s.pt")
    Trainer(DualBranchNet(cfg["model"]), tc, cfg["model"], str(tmp_path)).fit(loader, loader, verbose=False, state_path=state)
    other = {**tc, "epochs": 4}
    ran = []
    t = Trainer(DualBranchNet(cfg["model"]), other, cfg["model"], str(tmp_path))
    real = t.train_epoch
    t.train_epoch = lambda loader_, label, verbose: (ran.append(label), real(loader_, label, verbose))[1]
    hist = t.fit(loader, loader, verbose=False, state_path=state)
    assert len(ran) == 4 and hist["epochs_run"] == 4         # started at epoch 1 again
