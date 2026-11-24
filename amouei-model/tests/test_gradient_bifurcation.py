import torch

from networks.dual_branch_net import DualBranchNet


def test_both_branches_receive_gradients(cfg):
    """After loss.backward(), the CNN, the LSTM and the head all have gradients: the fusion head
    really uses both branches and the gradient splits at the concatenation point."""
    mcfg = cfg["model"]
    for block in mcfg.values():
        block["dropout"] = 0.0
    d = cfg["derived"]
    model = DualBranchNet(mcfg)
    out = model(torch.randn(8, d["C"], 4, 4), torch.randn(8, d["T"], d["F"]))
    torch.nn.functional.mse_loss(out, torch.randn(8, 1)).backward()

    for name, module in [("spatial", model.spatial), ("temporal", model.temporal), ("head", model.head)]:
        grads = [p.grad for p in module.parameters()]
        assert all(g is not None for g in grads), f"{name} has parameters without a gradient"
        assert sum(float(g.abs().sum()) for g in grads) > 0, f"{name} gradients are all zero"


def test_gradients_also_reach_the_cnn_with_flatten(cfg):
    mcfg = cfg["model"]
    for block in mcfg.values():
        block["dropout"] = 0.0
    mcfg["spatial_cnn"].update(feature_map="flatten", patch_px=4)
    d = cfg["derived"]
    model = DualBranchNet(mcfg)
    out = model(torch.randn(8, d["C"], 4, 4), torch.randn(8, d["T"], d["F"]))
    torch.nn.functional.mse_loss(out, torch.randn(8, 1)).backward()
    for module in (model.spatial, model.temporal, model.head):
        assert sum(float(p.grad.abs().sum()) for p in module.parameters()) > 0
