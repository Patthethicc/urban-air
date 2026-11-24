"""Progress, hang detection and a time budget for the long stages (training, tuning, inference).

On a CPU one training epoch takes minutes to tens of minutes and prints nothing until it ends, which looks exactly like a
freeze. This module
  1. prints a progress line inside long passes (`Progress`), with the time left,
  2. detects a REAL hang: a guarded loop that finishes no batch for `stall_timeout_s` is interrupted with a StageTimeout,
  3. optionally stops a command that runs past `max_hours`.
Limits come from the `runtime:` block of configs/train.yaml. A slow run is not an error; only the absence of progress is.

How a hang ends: the watchdog thread flags the main thread, which raises StageTimeout the moment it runs Python code again
(a clean stop; the run is marked timed_out). A thread stuck INSIDE a native call (a torch op, a swapped-out memory page,
a blocked read) never gets there, because the flag is only read between Python instructions; after `hard_exit_grace_s` the
process is therefore ended with a traceback and exit code 124. So a real hang costs stall_timeout_s (+ grace) at most.
"""
import _thread
import faulthandler
import os
import sys
import threading
import time
from contextlib import contextmanager

EXIT_CODE = 124                          # the code GNU `timeout` uses

_DEFAULTS = {"progress_every_s": 30, "status_every_s": 300, "stall_timeout_s": 600, "hard_exit_grace_s": 60,
             "max_hours": None, "warn_hours": 2}
_CFG = dict(_DEFAULTS)
_STARTED = _LAST_STATUS = time.monotonic()
_STACK: list = []                        # guarded loops currently running (innermost last)
_PLAN: dict = {}
_ANNOUNCED = False
_ON_HARD_EXIT = None


class StageTimeout(RuntimeError):
    """A stage stopped making progress, or ran past its time budget."""


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds < 90:
        return f"{seconds} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def configure(runtime: dict | None = None, on_hard_exit=None) -> None:
    """Apply the `runtime:` settings and start the time-budget clock. `on_hard_exit()` runs just before a forced exit."""
    global _STARTED, _LAST_STATUS, _ANNOUNCED, _ON_HARD_EXIT, _PLAN
    unknown = set(runtime or {}) - set(_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown train.runtime keys {sorted(unknown)}; allowed: {sorted(_DEFAULTS)}")
    _CFG.update(_DEFAULTS)
    _CFG.update(runtime or {})
    for key in ("progress_every_s", "status_every_s", "stall_timeout_s", "hard_exit_grace_s", "warn_hours"):
        if _CFG[key] is not None and float(_CFG[key]) < 0:
            raise ValueError(f"train.runtime.{key} must be >= 0 (0 switches it off)")
    if _CFG["max_hours"] is not None and float(_CFG["max_hours"]) <= 0:
        raise ValueError("train.runtime.max_hours must be > 0 or null")
    _STARTED = _LAST_STATUS = time.monotonic()
    _ANNOUNCED, _ON_HARD_EXIT, _PLAN = False, on_hard_exit, {}


def status_every_s() -> float:
    return float(_CFG["status_every_s"] or 0)


def set_plan(passes: float, max_epochs: int, typical_epochs: int = 12) -> None:
    """Tell the first epoch how many epoch-equivalents the run has, so it can print how long the whole run will take."""
    global _ANNOUNCED
    _PLAN.update(passes=passes, max_epochs=max_epochs, typical_epochs=typical_epochs)
    _ANNOUNCED = False


def check_budget() -> None:
    hours = _CFG["max_hours"]
    if hours is not None and time.monotonic() - _STARTED > float(hours) * 3600:
        raise StageTimeout(f"time budget of {hours} h used up (train.runtime.max_hours). Raise it or set it to null; "
                           "to shorten the run use a larger train.batch_size, fewer train.epochs or train.cv.enabled=false")


def _maybe_status(where: str) -> None:
    """For fits that run quietly (tuning): at most one 'still running, here is where' line per `status_every_s`."""
    global _LAST_STATUS
    every = _CFG["status_every_s"]
    now = time.monotonic()
    if every and where and now - _LAST_STATUS >= float(every):
        _LAST_STATUS = now
        print(f"  ... {where}  (running for {format_duration(now - _STARTED)})", flush=True)


class _Watch:
    """A thread that raises StageTimeout in the main thread when beat() has not been called for `stall` seconds."""

    def __init__(self, label: str, stall: float, grace: float, quiet: bool = False):
        self.label, self.stall, self.grace, self.quiet = label, float(stall), float(grace), quiet
        self.last, self.where, self.fired = time.monotonic(), "", None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"watchdog[{label}]", daemon=True)

    def beat(self, where: str = "") -> None:
        self.last = time.monotonic()
        if where:
            self.where = where
        check_budget()
        if self.quiet:
            _maybe_status(where)

    def _run(self) -> None:
        poll = min(1.0, max(0.05, self.stall / 10))
        previous = time.monotonic()
        while not self._stop.wait(poll):
            now = time.monotonic()
            if now - previous > max(30 * poll, 30.0):        # this thread itself was frozen: the computer slept
                self.last = now                              # so restart the clock instead of calling it a hang
            previous = now
            if now - self.last < self.stall:
                continue
            self.fired = (now - self.last, self.where)
            _thread.interrupt_main()                          # raises KeyboardInterrupt in the main thread...
            if not self._stop.wait(self.grace):               # ...unless it is stuck inside native code
                sys.stderr.write(f"\nTIMEOUT: {self.label}: no progress for {self.stall:.0f} s and Python cannot be "
                                 f"interrupted (stuck in native code); exiting. Where: {self.where or 'unknown'}\n")
                faulthandler.dump_traceback(all_threads=True)
                sys.stderr.flush()
                if _ON_HARD_EXIT:
                    try:
                        _ON_HARD_EXIT()
                    except Exception:
                        pass
                os._exit(EXIT_CODE)
            return


class _Idle:
    """The guard when stall detection is off: beats still enforce the time budget."""
    fired = None

    def __init__(self, quiet: bool = False):
        self.quiet = quiet

    def beat(self, where: str = "") -> None:
        check_budget()
        if self.quiet:
            _maybe_status(where)


@contextmanager
def guarded(label: str, quiet: bool = False):
    """Arm hang detection around one loop; call .beat() after every batch. A hang raises StageTimeout (or, stuck in
    native code, ends the process with exit code 124 after the grace period) instead of waiting forever.
    `quiet` loops (tuning fits print nothing per epoch) also give a position line every status_every_s."""
    stall = _CFG["stall_timeout_s"]
    watch = _Watch(label, stall, _CFG["hard_exit_grace_s"], quiet) if stall else _Idle(quiet)
    if stall:
        watch._thread.start()
    _STACK.append(watch)
    try:
        yield watch
    except KeyboardInterrupt:
        if watch.fired:                                        # our own interrupt; a real Ctrl+C passes through
            idle, where = watch.fired
            raise StageTimeout(
                f"{label}: no progress for {idle:.0f} s (train.runtime.stall_timeout_s = {stall}); last position: "
                f"{where or 'unknown'}. Training is normally one batch every fraction of a second, so this is a hang "
                "(low memory? a stuck process?). If one batch really takes this long (huge batch_size or "
                "inference.chunk_cells), raise train.runtime.stall_timeout_s.") from None
        raise
    finally:
        _STACK.pop()
        if stall:
            watch._stop.set()


def beat(where: str = "") -> None:
    """Signal progress to the innermost guarded loop (a no-op outside one apart from the time budget)."""
    if _STACK:
        _STACK[-1].beat(where)
    else:
        check_budget()


class Progress:
    """Progress lines for one long pass: an estimate once it is clear the pass is slow, then one line per interval."""

    def __init__(self, total: int, label: str, enabled: bool = True, announce: bool = False, unit: str = "batches"):
        self.total, self.label, self.unit, self.announce = int(total), label, unit, announce
        self.every = float(_CFG["progress_every_s"] or 0)
        self.enabled = bool(enabled) and self.every > 0 and self.total > 0
        self.started = self._last = time.monotonic()
        self.estimated = False

    def update(self, done: int) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        elapsed = now - self.started
        if elapsed < 1 or done <= 0:
            return
        eta_total = elapsed / done * self.total
        if self.announce and not self.estimated and (done >= 20 or elapsed >= self.every):
            self.estimated = True
            if eta_total >= self.every:
                print(f"{self.label}: {self.total:,} {self.unit}, about {format_duration(eta_total)} for this pass at the "
                      f"current speed; a progress line follows every {format_duration(self.every)}", flush=True)
                self._project(eta_total)
            self._last = now
        elif now - self._last >= self.every:
            self._last = now
            print(f"{self.label}: {done / self.total:5.1%}  {done:,}/{self.total:,} {self.unit}  "
                  f"elapsed {format_duration(elapsed)}  remaining ~{format_duration(eta_total - elapsed)}", flush=True)

    @staticmethod
    def _project(epoch_seconds: float) -> None:
        """Once per run: what the whole run costs at this epoch time (needs set_plan)."""
        global _ANNOUNCED
        if _ANNOUNCED or not _PLAN:
            return
        _ANNOUNCED = True
        typical, worst = (epoch_seconds * _PLAN["passes"] * n for n in (_PLAN["typical_epochs"], _PLAN["max_epochs"]))
        slow = typical / 3600 > float(_CFG["warn_hours"] or 1e9)
        print(f"{'WARNING: ' if slow else ''}the whole run takes about {format_duration(typical)} if every fit stops after "
              f"~{_PLAN['typical_epochs']} epochs (early stopping), at most {format_duration(worst)}"
              f"{'. A larger train.batch_size shortens it (a batch costs almost the same at 64 or 1024)' if slow else ''}",
              flush=True)
