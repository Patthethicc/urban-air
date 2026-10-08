"""Plot functions. Each takes data + an output path and saves a PNG."""


def plot_loss_curve(history: dict, out_path: str) -> None:
    raise NotImplementedError


def plot_pred_vs_obs(y_true, y_pred, out_path: str) -> None:
    raise NotImplementedError


def plot_residual_hist(y_true, y_pred, out_path: str) -> None:
    raise NotImplementedError


def plot_timeseries_sample(y_true, y_pred, out_path: str) -> None:
    raise NotImplementedError
