import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

import run_pipeline
import runtime_guard as rg
from networks.dual_branch_net import DualBranchNet
from runtime_guard import Progress, StageTimeout, beat, configure, guarded
from training.trainer import Trainer

MODEL_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _fresh_limits():
    configure({})
    yield
    configure({})


def _toy(cfg, n=64):
    d = cfg["derived"]
    g = torch.Generator().manual_seed(0)
    return TensorDataset(torch.randn(n, d["C"], 2, 2, generator=g), torch.randn(n, d["T"], d["F"], generator=g),
                         torch.randn(n, 1, generator=g))


# ------------------------------------------------------------------------------ hang detection
def _hang(seconds=30.0):
    """A hang in Python code (it keeps returning to the interpreter, so the watchdog's flag can land)."""
    for _ in range(int(seconds / 0.05)):
        time.sleep(0.05)


def test_a_loop_that_stops_making_progress_is_interrupted_with_a_clear_error():
    configure({"stall_timeout_s": 0.5})
    started = time.monotonic()
    with pytest.raises(StageTimeout, match=r"no progress for .*last position: batch 3"):
        with guarded("toy loop") as g:
            for i in range(10):
                g.beat(f"batch {i}")
                time.sleep(0.05) if i < 3 else _hang()            # batch 3 hangs
    assert time.monotonic() - started < 10                         # stopped by the watchdog, not after the 30 s


def test_a_loop_that_keeps_beating_is_never_interrupted_however_long_it_runs():
    configure({"stall_timeout_s": 0.5})
    with guarded("busy loop") as g:
        for _ in range(25):
            time.sleep(0.1)                                       # 2.5 s in total, a beat every 0.1 s
            g.beat()


def test_a_real_ctrl_c_is_not_turned_into_a_timeout():
    configure({"stall_timeout_s": 60})
    import _thread
    with pytest.raises(KeyboardInterrupt):
        with guarded("loop"):
            _thread.interrupt_main()
            time.sleep(5)


def test_stall_detection_can_be_switched_off():
    configure({"stall_timeout_s": 0})
    with guarded("loop") as g:
        time.sleep(0.8)
        g.beat()


def test_the_time_budget_stops_a_run_at_the_next_batch():
    configure({"max_hours": 1e-6})                                 # 3.6 ms
    time.sleep(0.05)
    with pytest.raises(StageTimeout, match="time budget of 1e-06 h"):
        with guarded("loop") as g:
            g.beat()
    with pytest.raises(StageTimeout, match="time budget"):
        beat()                                                    # outside a guarded loop too


def test_a_process_stuck_in_a_blocking_call_is_ended_with_a_traceback_and_code_124(tmp_path):
    """The flag cannot wake a thread inside one long native call (time.sleep stands in for a stuck torch op / read),
    so after the grace period the process leaves by itself instead of hanging for ever."""
    script = tmp_path / "hang.py"
    script.write_text(textwrap.dedent(f"""
        import sys
        sys.path.insert(0, r"{MODEL_DIR}")
        import time
        import runtime_guard as rg
        rg.configure({{"stall_timeout_s": 1, "hard_exit_grace_s": 1}})
        try:
            with rg.guarded("native hang") as g:
                g.beat("before the hang")
                time.sleep(600)
        except rg.StageTimeout:
            print("interrupted cleanly")        # a platform whose sleep the flag does wake
            sys.exit(0)
        print("unreachable")
    """), encoding="utf-8")
    started = time.monotonic()
    r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=90,
                       stdin=subprocess.DEVNULL)
    assert time.monotonic() - started < 60 and "unreachable" not in r.stdout          # not the 600 s of the hang
    if r.returncode == 0:
        assert "interrupted cleanly" in r.stdout
    else:
        assert r.returncode == rg.EXIT_CODE == 124
        assert "TIMEOUT: native hang" in r.stderr and "before the hang" in r.stderr


def test_bad_runtime_settings_are_rejected():
    with pytest.raises(ValueError, match="unknown train.runtime keys"):
        configure({"stall_timeout": 5})                           # typo
    with pytest.raises(ValueError, match="max_hours"):
        configure({"max_hours": 0})
    with pytest.raises(ValueError, match=">= 0"):
        configure({"stall_timeout_s": -1})


# ------------------------------------------------------------------------------ progress lines
def test_a_fast_pass_prints_nothing(monkeypatch, capsys):
    configure({"progress_every_s": 30})
    p = Progress(100, "[fold 0] epoch 1/50", announce=True)
    p.started = p._last = 0.0
    for i in range(1, 101):
        monkeypatch.setattr(rg.time, "monotonic", lambda i=i: i * 0.01)          # the whole pass takes 1 s
        p.update(i)
    assert capsys.readouterr().out == ""


def test_a_slow_pass_announces_its_length_once_then_prints_a_line_per_interval(monkeypatch, capsys):
    configure({"progress_every_s": 30})
    p = Progress(1000, "[fold 0] epoch 1/50", announce=True)
    p.started = p._last = 0.0
    for i in range(1, 1001):
        monkeypatch.setattr(rg.time, "monotonic", lambda i=i: i * 1.0)           # one second per batch: a 1000 s pass
        p.update(i)
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("[fold 0] epoch 1/50: 1,000 batches, about 17 min for this pass")
    lines = [x for x in out if "remaining" in x]
    assert 25 <= len(lines) <= 40 and "50.0%" in " ".join(lines)                # about one line per 30 s


def test_the_first_epoch_says_how_long_the_whole_run_takes_and_warns_when_it_is_long(monkeypatch, capsys):
    configure({"progress_every_s": 30, "warn_hours": 2})
    rg.set_plan(passes=4.5, max_epochs=50)
    p = Progress(100, "[fold 0] epoch 1/50", announce=True)
    p.started = p._last = 0.0
    monkeypatch.setattr(rg.time, "monotonic", lambda: 1116.0)                   # 18.6 min per epoch
    p.update(20)                                                                # 5580 s for 100 batches at this rate
    out = capsys.readouterr().out
    assert "WARNING: the whole run takes about" in out and "larger train.batch_size" in out
    p2 = Progress(100, "again", announce=True)
    p2.started = p2._last = 0.0
    p2.update(20)
    assert capsys.readouterr().out.count("the whole run takes") == 0            # announced once per run


def test_progress_can_be_switched_off(capsys):
    configure({"progress_every_s": 0})
    p = Progress(10, "x", announce=True)
    p.started = p._last = time.monotonic() - 1000
    p.update(5)
    assert capsys.readouterr().out == ""


# ------------------------------------------------------------------------------ in the real trainer
class _Slow(Dataset):
    def __init__(self, base, slow_from):
        self.base, self.slow_from = base, slow_from

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        if i >= self.slow_from:
            _hang()                                                             # a data loader that hangs
        return self.base[i]


def test_the_trainer_stops_on_a_hung_batch_and_names_the_epoch(cfg, tmp_path):
    configure({"stall_timeout_s": 3})
    tc = {**cfg["train"], "device": "cpu", "batch_size": 16}
    trainer = Trainer(DualBranchNet(cfg["model"]), tc, cfg["model"], str(tmp_path))
    loader = DataLoader(_Slow(_toy(cfg, 64), slow_from=32), batch_size=16)
    started = time.monotonic()
    with pytest.raises(StageTimeout, match=r"\[fold 0\] epoch 1/5.*batch 2/4"):
        trainer.fit(loader, None, epochs=5, verbose=False, tag="[fold 0]")
    assert time.monotonic() - started < 20


class _SlowButAlive(Dataset):
    """Every sample takes 40 ms: a batch of 4 takes 0.16 s, the 8-batch epoch 1.3 s, longer than the 0.6 s stall limit."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        time.sleep(0.04)
        return self.base[i]


def test_a_slow_but_healthy_epoch_is_not_called_a_hang(cfg, tmp_path):
    """Only the ABSENCE of progress counts: an epoch far longer than the limit is fine while batches keep finishing."""
    configure({"stall_timeout_s": 0.6})
    tc = {**cfg["train"], "device": "cpu", "early_stopping_patience": 100}
    trainer = Trainer(DualBranchNet(cfg["model"]), tc, cfg["model"], str(tmp_path))
    started = time.monotonic()
    hist = trainer.fit(DataLoader(_SlowButAlive(_toy(cfg, 32)), batch_size=4), None, epochs=1, verbose=False)
    assert hist["epochs_run"] == 1 and time.monotonic() - started > 1.0           # it really did outlast the limit


def test_a_healthy_trainer_run_is_unchanged_by_the_guard(cfg, tmp_path):
    configure({"stall_timeout_s": 5})
    tc = {**cfg["train"], "device": "cpu", "early_stopping_patience": 100}
    trainer = Trainer(DualBranchNet(cfg["model"]), tc, cfg["model"], str(tmp_path))
    loader = DataLoader(_toy(cfg, 64), batch_size=16)
    hist = trainer.fit(loader, loader, epochs=3, verbose=False)
    assert hist["epochs_run"] == 3 and len(hist["val_loss"]) == 3


# ------------------------------------------------------------------------------ in the CLI
def _run_dir(cfg, tmp_path, with_model=False):
    ckpts = tmp_path / "ckpts"
    d = ckpts / "20260101_000000"
    d.mkdir(parents=True)
    (d / "run_info.json").write_text(json.dumps({"run_id": d.name, "status": "running"}), encoding="utf-8")
    if with_model:
        (d / run_pipeline.MODEL_FILE).write_bytes(b"x")
    cfg["data"]["paths"]["checkpoints_dir"] = str(ckpts)
    return d


def test_a_timeout_marks_the_unfinished_run_in_its_run_info(cfg, tmp_path):
    d = _run_dir(cfg, tmp_path)
    run_pipeline._record_timeout(cfg, "train", "no progress for 600 s")
    info = json.loads((d / "run_info.json").read_text(encoding="utf-8"))
    assert info["status"] == "timed_out" and info["timeout"] == "no progress for 600 s"


def test_a_finished_run_is_never_marked_and_other_stages_are_left_alone(cfg, tmp_path):
    done = _run_dir(cfg, tmp_path, with_model=True)
    run_pipeline._record_timeout(cfg, "train", "x")
    run_pipeline._record_timeout(cfg, "infer", "x")
    assert json.loads((done / "run_info.json").read_text(encoding="utf-8"))["status"] == "running"


def test_the_cli_exits_with_code_124_and_a_message_on_a_timeout(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--stage", "coverage"])
    monkeypatch.setattr(run_pipeline, "run", lambda cfg, args: (_ for _ in ()).throw(StageTimeout("no progress for 600 s")))
    with pytest.raises(SystemExit) as e:
        run_pipeline.main()
    assert e.value.code == 124
    assert "TIMEOUT in stage 'coverage': no progress for 600 s" in capsys.readouterr().err


def test_the_shipped_defaults_are_the_documented_ones(base_cfg):
    r = base_cfg["train"]["runtime"]
    assert r == {"progress_every_s": 30, "status_every_s": 300, "stall_timeout_s": 600, "hard_exit_grace_s": 60,
                 "max_hours": None, "warn_hours": 2}
    configure(r)                                                                # and they pass validation


def test_format_duration_is_still_importable_from_the_inference_module():
    from inference.cell_inference import format_duration
    assert format_duration(30) == "30 s" and format_duration(600) == "10 min" and format_duration(36000) == "10.0 h"
