"""Write metrics_summary.json and plots into snapshots/<run_id>/."""


def generate_snapshot(run_id: str, cfg: dict) -> None:
    """Load best checkpoint, predict on test, inverse-scale, compute metrics,
    write metrics_summary.json, call plotters."""
    # TODO
    raise NotImplementedError
