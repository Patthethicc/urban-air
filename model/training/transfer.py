"""Transfer learning for the DualBranchNet: start from the weights of another model, optionally keep parts fixed.

train.init_from : path of a checkpoint (.pt) whose weights are copied into the new model before training. A layer is copied only when its
                  name AND shape match, so a model pretrained WITHOUT the CNN (the temporal branch only, see transfer/pretrain.py) fills the
                  temporal branch of a CNN-LSTM, while the fusion head (a different input size) and the CNN stay at their fresh values.
train.freeze    : parts that are NOT trained: any of "spatial" (the CNN), "temporal" (the LSTM), "head" (the fusion head).
The temporal branch must load completely when init_from is set: a half-loaded LSTM means the two models do not share the input features
(another lag count, another feature set) and the transfer would silently do nothing.
"""
from __future__ import annotations

import torch

PARTS = ("spatial", "temporal", "head")
last_report: dict = {}          # what the most recent build_model did; train() stores it in run_info.json


def warm_start(model: torch.nn.Module, path: str) -> dict:
    """Copy the matching layers of the checkpoint at `path` into `model`. Returns a report; raises if the temporal branch does not match."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    source, target = ckpt["model_state"], model.state_dict()
    loaded, skipped = [], []
    for name, tensor in source.items():
        if name in target and target[name].shape == tensor.shape:
            target[name] = tensor.clone()
            loaded.append(name)
        else:
            skipped.append(name)
    temporal_all = [k for k in target if k.startswith("temporal.")]
    temporal_missing = [k for k in temporal_all if k not in loaded]
    if temporal_missing:
        raise ValueError(f"init_from {path}: the temporal branch does not match (not loaded: {temporal_missing[:4]}...). The pretrained model must have "
                         "the same lags, features and LSTM size (model.temporal_lstm) as this one.")
    model.load_state_dict(target)
    return {"path": str(path), "layers_loaded": len(loaded), "layers_skipped": skipped,
            "parts_loaded": sorted({k.split(".")[0] for k in loaded}), "source_epoch": ckpt.get("epoch"), "source_val_loss": ckpt.get("val_loss")}


def freeze_parts(model: torch.nn.Module, parts) -> dict:
    """requires_grad = False for the named parts. A part the model does not have (the CNN of an LSTM-only model) is ignored. Returns counts."""
    unknown = [p for p in parts if p not in PARTS]
    if unknown:
        raise ValueError(f"train.freeze has unknown part(s) {unknown}; use any of {list(PARTS)}")
    frozen = []
    for part in parts:
        module = getattr(model, part, None)
        if module is None:
            continue
        for p in module.parameters():
            p.requires_grad = False
        frozen.append(part)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not trainable:
        raise ValueError("train.freeze leaves nothing to train: every part of the model is frozen")
    return {"frozen": frozen, "trainable_parameters": int(trainable)}


def build_model(model_cfg: dict, train_cfg: dict):
    """DualBranchNet from the config, warm-started and frozen as train.init_from / train.freeze say (both default to nothing)."""
    from networks.dual_branch_net import DualBranchNet

    model = DualBranchNet(model_cfg)
    report: dict = {}
    if train_cfg.get("init_from"):
        report["init"] = warm_start(model, train_cfg["init_from"])
    if train_cfg.get("freeze"):
        report["freeze"] = freeze_parts(model, train_cfg["freeze"])
    if report and report != last_report:
        init = report.get("init")
        if init:
            print(f"transfer: {init['layers_loaded']} layers loaded from {init['path']} (parts {init['parts_loaded']}); "
                  f"{len(init['layers_skipped'])} left at their fresh values")
        if "freeze" in report:
            print(f"transfer: frozen {report['freeze']['frozen']}, {report['freeze']['trainable_parameters']:,} trainable parameters")
    last_report.clear()
    last_report.update(report)
    return model
