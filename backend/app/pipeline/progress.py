"""One progress display for the whole pipeline.

Pipeline code only calls Bar / ok / warn. The worker registers a hook with set_hook()
to turn those events into job progress; without a hook everything just prints.
"""
import os
import warnings
from contextlib import contextmanager

from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.text import Text

console = Console()
_progress = None
_hook = None        # callable(event, **data); a plain global so it also works from threads


def quiet_noise() -> None:
    """Call before importing mediapipe / torch code."""
    os.environ.setdefault("GLOG_minloglevel", "3")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    warnings.filterwarnings("ignore", message=r"std\(\): degrees of freedom")


def set_hook(hook):
    """Register callback(event, **data), or None to clear it. Returns the previous hook."""
    global _hook
    previous, _hook = _hook, hook
    return previous


def _emit(event: str, **data) -> None:
    if _hook is None:
        return
    try:
        _hook(event, **data)
    except Exception as exc:
        # A cancelled job must stop the pipeline. Any other hook error must never break it.
        if type(exc).__name__ == "JobCanceled":
            raise


def _ensure():
    global _progress
    if _progress is None:
        _progress = Progress(
            TextColumn("{task.description:<26}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        )
        _progress.start()
    return _progress


def stop() -> None:
    global _progress
    if _progress is not None:
        _progress.stop()
        _progress = None


@contextmanager
def pipeline(hook=None):
    """Optional wrapper for scripts: closes the display even on errors."""
    previous = set_hook(hook) if hook is not None else _hook
    _ensure()
    try:
        yield
    finally:
        set_hook(previous)
        stop()


# ---------- messages (use these instead of print) ----------
def log(msg, style="") -> None:
    text = str(msg)
    console.print(Text(text, style=style))     # Text() so "[youtube]" isn't read as markup
    _emit("log", message=text, style=style)


def info(msg):  log(f"• {msg}", "cyan")
def ok(msg):    log(f"✓ {msg}", "green")
def warn(msg):  log(f"! {msg}", "yellow")
def error(msg): log(f"✗ {msg}", "bold red")


# ---------- bars ----------
class Bar:
    """One line in the display. total=None gives a pulsing bar for work with no progress info."""

    def __init__(self, name, total=1.0):
        self._p = _ensure()
        self.name = name
        self.total = total
        self.id = self._p.add_task(name, total=total)
        _emit("bar_start", name=name, total=total)

    def update(self, completed, total=None):
        kw = {"completed": completed}
        if total:
            self.total = total
            kw["total"] = total
        self._p.update(self.id, **kw)
        _emit("bar_update", name=self.name, completed=completed, total=self.total)

    def finish(self):
        if self.total is None:
            self.total = 1
        self._p.update(self.id, completed=self.total, total=self.total)
        _emit("bar_finish", name=self.name, total=self.total)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        if exc_type is None:
            self.finish()


def track(items, name, total=None):
    """for seg in track(segments, "label"): ...   (the bar updates itself)"""
    if total is None:
        total = len(items) if hasattr(items, "__len__") else None
    bar = Bar(name, total)
    for i, item in enumerate(items, 1):
        yield item
        bar.update(i)
    bar.finish()


# ---------- library adapters ----------
class DiarizationHook:
    """Replacement for pyannote's ProgressHook."""

    def __init__(self):
        self.bars = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for bar in self.bars.values():
            bar.finish()

    def __call__(self, step_name, step_artifact, file=None, total=None, completed=None):
        if completed is None:
            completed = total = 1
        if step_name not in self.bars:
            self.bars[step_name] = Bar(step_name, total)
        self.bars[step_name].update(completed, total)


def ytdlp_hook():
    """ydl_opts['progress_hooks'] = [ytdlp_hook()]   (one bar per stream)"""
    bars = {}

    def hook(d):
        key = d.get("filename", "")
        if d["status"] == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            if key not in bars:
                bars[key] = Bar(f"download: stream {len(bars) + 1}", total or None)
            bars[key].update(d.get("downloaded_bytes", 0), total)
        elif d["status"] == "finished" and key in bars:
            bars[key].finish()

    return hook
