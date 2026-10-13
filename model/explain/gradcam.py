"""Grad-CAM on the CNN branch (Output.txt): which pixels of a morphology patch drove a prediction.

For a patch x and the scalar PM2.5 output y, with A_k the k-th channel of the LAST convolutional layer:
    alpha_k = mean over pixels of dy/dA_k        (how much channel k matters for y)
    CAM     = ReLU( sum_k alpha_k * A_k )         (where those channels are active)
The map is rescaled to [0, 1] per patch. Convolution padding keeps its size equal to the patch, so it
overlays the input directly. The LSTM branch is part of the forward pass (the gradient is of the full
prediction), so the saliency is "what in this patch mattered given this temporal history".
"""
import numpy as np
import torch


def grad_cam(model, patches: np.ndarray, temporal: np.ndarray, device: str = "cpu"):
    """patches [n, C, H, W] and temporal [n, T, F] (both scaled like in training).
    Returns (cam [n, H, W] in [0, 1], prediction [n] in scaled units)."""
    if getattr(model, "spatial", None) is None:
        raise ValueError("the model has no CNN branch (model.spatial_cnn.enabled is false): nothing to explain")
    model = model.to(device).eval()
    store: dict = {}

    def keep_activation(_module, _inputs, output):
        output.retain_grad()
        store["a"] = output

    handle = model.spatial.conv[-1].register_forward_hook(keep_activation)
    try:
        with torch.enable_grad():
            model.zero_grad()
            pred = model(torch.from_numpy(np.ascontiguousarray(patches)).to(device),
                         torch.from_numpy(np.ascontiguousarray(temporal)).to(device))
            pred.sum().backward()          # samples are independent in eval mode, so one backward serves all
    finally:
        handle.remove()

    act, grad = store["a"].detach(), store["a"].grad.detach()
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * act).sum(dim=1))                          # [n, H, W]
    peak = cam.flatten(1).max(dim=1).values.clamp_min(1e-12).view(-1, 1, 1)
    return (cam / peak).cpu().numpy(), pred.detach().cpu().numpy().ravel()


def attention_profile(raw_patches: np.ndarray, cam: np.ndarray, band_names: list[str], quantile: float = 0.8) -> list[dict]:
    """For each patch and band: mean raster value on the ATTENDED pixels (CAM >= its `quantile`, and > 0)
    versus the whole patch, in patch-standard-deviation units (positive = the model looks at the higher
    values of that band, e.g. tall buildings, dense blocks, wide streets, deep canyons)."""
    rows = []
    for i in range(len(cam)):
        c = cam[i]
        mask = (c >= np.quantile(c, quantile)) & (c > 0)
        if not mask.any():
            continue
        for b, name in enumerate(band_names[:raw_patches.shape[1]]):
            band = raw_patches[i, b]
            std = float(band.std())
            rows.append({"patch": i, "band": name, "attended_mean": float(band[mask].mean()),
                         "patch_mean": float(band.mean()), "attended_pixels": int(mask.sum()),
                         "std_units": float((band[mask].mean() - band.mean()) / std) if std > 1e-9 else 0.0})
    return rows
