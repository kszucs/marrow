"""A live display for a command that runs for minutes and says nothing.

Separate from `devkit.mojo` so that module needs nothing but the standard
library.  That is not tidiness: `python/build.py` is a hatchling build hook, and
the environment cibuildwheel creates for it holds `hatchling` and the Mojo
compiler and nothing else -- no `rich`, no `psutil`.  Keeping the display here
is what lets the wheel build share the one definition of the shared-library
recipe instead of carrying a second copy of it.

`SilentProgress` in `devkit.mojo` is the other implementation of this protocol,
and the default: `start` / `attach` / `finish` / `peak_rss`.
"""

import sys

import psutil
from rich.console import Console
from rich.progress import Progress as RichProgress
from rich.progress import ProgressColumn, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.text import Text


class _MemoryColumn(ProgressColumn):
    """Resident size of the tracked process, sampled when the display renders.

    Elaboration is the slow part of a Mojo compile and its cost shows up as
    resident memory, so live RSS is the most informative thing to show while
    waiting minutes for a single unit.  Sampling from the renderer rather than
    from a thread of its own means the sample rate *is* the refresh rate, and
    there is nothing to join on shutdown.
    """

    def __init__(self):
        super().__init__()
        self.peak = 0
        self._process = None

    def attach(self, pid):
        try:
            self._process = psutil.Process(pid)
        except psutil.Error:
            self._process = None

    def render(self, task):
        rss = 0
        if self._process is not None:
            try:
                rss = self._process.memory_info().rss
            except psutil.Error:
                self._process = None
        if not rss:
            return Text("")
        self.peak = max(self.peak, rss)
        return Text(f"{rss / 1e9:.1f} GB", style="progress.data.speed")


class ConsoleProgress:
    """A live spinner, elapsed time and RSS while a command runs.

    A single Mojo compile runs for minutes with no output of its own, which is
    indistinguishable from a hang.  On a terminal this shows a live line; off
    one it prints a plain start/finish pair so CI logs still show what happened.

    `isatty()` has to be asked inside `start`, not at construction: pytest
    captures at the file-descriptor level, so until capture is suspended fd 2
    points at a temp file and every terminal looks non-interactive.
    """

    def __init__(self):
        self._console = None
        self._progress = None
        self._memory = None

    @property
    def peak_rss(self):
        return self._memory.peak if self._memory is not None else 0

    def start(self, label):
        stream = sys.__stderr__
        self._console = Console(file=stream, highlight=False)
        if stream is None or not stream.isatty():
            self._console.print(f"{label} ...")
            return
        # Start on a line of our own -- pytest is mid-way through writing its
        # per-file progress line, and appending to it reads as garbage.
        self._console.print()
        self._memory = _MemoryColumn()
        self._progress = RichProgress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TextColumn("—"),
            TimeElapsedColumn(),
            self._memory,
            console=self._console,
            transient=True,
        )
        self._progress.start()
        self._progress.add_task(label, total=None)

    def attach(self, pid):
        if self._memory is not None:
            self._memory.attach(pid)

    def finish(self, label, result):
        """Tear the live region down, then say what happened.

        *result* is None when the command never produced one -- the executable
        was missing, say.  Tearing down still has to happen, or the live region
        outlives the run and swallows the rest of the session's output.
        """
        if self._progress is not None:
            self._progress.stop()
            self._progress = None
        if result is None:
            self._console.print(f"✗ {label} — did not run")
            return
        mark = "✓" if result.ok else "✗"
        memory = f", peak {result.peak_rss / 1e9:.1f} GB" if result.peak_rss else ""
        suffix = " (TIMED OUT)" if result.timed_out else ""
        self._console.print(f"{mark} {label} — {result.elapsed:.0f}s{memory}{suffix}")
