# Dual-branch (CNN + LSTM) PM2.5 model

Run from this folder:

    python run_pipeline.py --stage preprocess   # writes processed/*.npy + scalers
    python run_pipeline.py --stage train        # writes checkpoints/<run_id>/best_dual_branch_model.pt
    python run_pipeline.py --stage report       # writes snapshots/<run_id>/metrics_summary.json + plots
    pytest tests

Build order: configs/seed -> preprocessing -> networks -> training -> checkpoint/run_pipeline -> reporting.
