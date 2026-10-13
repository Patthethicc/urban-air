"""Save/load weights together with the hyperparameters that built the model."""
import hashlib
import json
from pathlib import Path

import torch


def file_sha1(path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def check_compatibility(ckpt: dict, processed_dir: str) -> None:
    """Calibration / test inference must use exactly what the model was trained with: patch extent and
    size, band count, sequence length (lags), feature order and the normalisation parameters.
    Raises ValueError listing every mismatch (e.g. preprocessing was re-run with other settings)."""
    extra = ckpt.get("extra") or {}
    derived = extra.get("derived")
    if not derived:                                   # checkpoint from before this check existed
        return
    processed = Path(processed_dir)
    info = json.loads((processed / "preprocess_info.json").read_text(encoding="utf-8"))
    problems = [f"{k}: trained with {derived[k]!r}, data has {info.get(k)!r}"
                for k in ("T", "F", "C", "patch_px", "feature_names") if derived.get(k) != info.get(k)]
    if extra.get("crs") and info.get("crs") and extra["crs"] != info["crs"]:
        problems.append(f"crs: trained in {extra['crs']}, data is in {info['crs']}")
    if "lags" in extra and extra["lags"] != info["temporal"]["lags"]:
        problems.append(f"lags: trained with {extra['lags']}, data has {info['temporal']['lags']}")
    if extra.get("scalers_sha1") and extra["scalers_sha1"] != file_sha1(processed / "scalers.json"):
        problems.append("scalers.json changed since training (normalisation parameters differ)")
    if problems:
        raise ValueError("checkpoint does not match the processed data:\n  " + "\n  ".join(problems))


def save_checkpoint(path: str, model, model_cfg: dict, train_cfg: dict, epoch: int, val_loss,
                    extra: dict | None = None) -> None:
    """Store state_dict + model_cfg + train_cfg + epoch + val_loss (+ extra, e.g. derived data shapes)."""
    torch.save({
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_cfg": model_cfg,
        "train_cfg": train_cfg,
        "epoch": int(epoch),
        "val_loss": None if val_loss is None else float(val_loss),
        "extra": extra or {},
        "torch_version": torch.__version__,
    }, path)


def load_model(path: str, device: str = "cpu"):
    """Rebuild DualBranchNet from the saved model_cfg, load weights, return (model in eval mode, checkpoint)."""
    from networks.dual_branch_net import DualBranchNet

    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = DualBranchNet(ckpt["model_cfg"])
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), ckpt
