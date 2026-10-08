"""Save/load weights together with the hyperparameters that built the model."""
import torch


def save_checkpoint(path: str, model, model_cfg: dict, train_cfg: dict,
                    epoch: int, val_loss: float) -> None:
    """Store state_dict + model_cfg + train_cfg + epoch + val_loss."""
    # TODO
    raise NotImplementedError


def load_model(path: str, device: str = "cpu"):
    """Rebuild DualBranchNet from the saved model_cfg, load weights, return model in eval mode."""
    # TODO
    raise NotImplementedError
