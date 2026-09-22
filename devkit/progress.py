"""A live display for a command that runs for minutes and says nothing.

Separate from `devkit.mojo` so that module needs nothing but the standard
library.  That is not tidiness: `python/build.py` is a hatchling build hook, and
the environment cibuildwheel creates for it holds `hatchling` and the Mojo
compiler and nothing else -- no `rich`, no `psutil`.  Keeping the display here
is what lets the wheel build share the one definition of the shared-library
recipe instead of carrying a second copy of it.

`SilentProgress` in `devkit.mojo` is the other implementation of this protocol,
and the default: `start` / `attach` / `finish` / `snapshot` / `peak_rss`.
"""

import io
import sys

import psutil
from rich.console import Console
from rich.progress import Progress as RichProgress
from rich.progress import ProgressColumn, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.text import Text


class _Usage:
    """Resident size and CPU time of the tracked process.

    Peak RSS is kept as it is seen rather than read at the end, because a
    process that has exited reports nothing: there is no total left to ask for.

    `snapshot` is the other half, and the reason this is not just a column.
    CPU time is what separates a unit that is merely slower than its deadline
    -- burning a core, CPU time tracking elapsed -- from one that has
    deadlocked, which burns nothing.  The deadline is the only moment that
    reading exists, since `ProcessRunner` kills the tree immediately
    afterwards, so it is taken there rather than guessed at from the log later.
    """

    def __init__(self):
        self.peak = 0
        self._process = None

    def attach(self, pid):
        try:
            self._process = psutil.Process(pid)
        except psutil.Error:
            self._process = None

    def sample(self):
        """Current RSS, with `peak` updated.  0 once the process is gone.

        The process rather than the tree: this runs at the display's refresh
        rate, and walking every child ten times a second costs more than the
        number is worth.  `snapshot` runs once, and does walk them.
        """
        if self._process is None:
            return 0
        try:
            rss = self._process.memory_info().rss
        except psutil.Error:
            self._process = None
            return 0
        self.peak = max(self.peak, rss)
        return rss

    def snapshot(self):
        """`(cpu_seconds, rss)` across the tree, or None if it cannot be read.

        The tree, because `mojo` spawns children, and a parent that spent the
        whole run waiting on one reads as no CPU at all -- which is the very
        reading this exists to make trustworthy.
        """
        if self._process is None:
            return None
        try:
            processes = [self._process] + self._process.children(recursive=True)
        except psutil.Error:
            return None
        cpu = 0.0
        rss = 0
        for process in processes:
            try:
                spent = process.cpu_times()
                cpu += spent.user + spent.system
                rss += process.memory_info().rss
            except psutil.Error:
                continue  # Exited mid-walk; what the rest report still counts.
        self.peak = max(self.peak, rss)
        return cpu, rss


class _MemoryColumn(ProgressColumn):
    """Resident size of the tracked process, sampled when the display renders.

    Elaboration is the slow part of a Mojo compile and its cost shows up as
    resident memory, so live RSS is the most informative thing to show while
    waiting minutes for a single unit.
    """

    def __init__(self, usage):
        super().__init__()
        self._usage = usage

    def render(self, task):
        rss = self._usage.sample()
        if not rss:
            return Text("")
        return Text(f"{rss / 1e9:.1f} GB", style="progress.data.speed")


class ConsoleProgress:
    """A live spinner, elapsed time and RSS while a command runs.

    A single Mojo compile runs for minutes with no output of its own, which is
    indistinguishable from a hang.  On a terminal this shows a live line; off
    one it prints a plain start/finish pair so CI logs still show what happened.

    Usage is tracked either way, and by the same means: rich's refresh thread
    evaluates the columns whether or not there is a terminal to draw them on,
    so sampling comes free with the display and the off-terminal case needs no
    second code path -- and no polling thread of our own.  It used to be
    tracked only on a terminal, which left the finish line's peak at zero in CI
    and a timeout there unattributable, the one place those numbers are the
    only evidence anyone gets.  `test_console_progress_samples_off_a_terminal`
    is what stops that lapsing back if rich ever skips the evaluation.

    `isatty()` has to be asked inside `start`, not at construction: pytest
    captures at the file-descriptor level, so until capture is suspended fd 2
    points at a temp file and every terminal looks non-interactive.
    """

    def __init__(self):
        self._console = None
        self._progress = None
        self._usage = None

    @property
    def peak_rss(self):
        return self._usage.peak if self._usage is not None else 0

    def snapshot(self):
        return self._usage.snapshot() if self._usage is not None else None

    def start(self, label):
        stream = sys.__stderr__
        self._console = Console(file=stream, highlight=False)
        self._usage = _Usage()
        if stream is None:
            return
        if stream.isatty():
            # Start on a line of our own -- pytest is mid-way through writing
            # its per-file progress line, and appending to it reads as garbage.
            self._console.print()
            display = self._console
        else:
            # Nothing is drawn off a terminal, so say what started; the finish
            # line is the other half of the pair a CI log gets.  The live
            # region still runs, because its refresh is what samples usage, but
            # it renders into a sink of its own: pointed at the real stream it
            # leaves a blank line per unit in the log.
            self._console.print(f"{label} ...")
            display = Console(file=io.StringIO(), highlight=False)
        self._progress = RichProgress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TextColumn("—"),
            TimeElapsedColumn(),
            _MemoryColumn(self._usage),
            console=display,
            transient=True,
        )
        self._progress.start()
        self._progress.add_task(label, total=None)

    def attach(self, pid):
        if self._usage is not None:
            self._usage.attach(pid)

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
