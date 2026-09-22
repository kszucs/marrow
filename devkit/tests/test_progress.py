"""What a command was spending while it said nothing.

The display is cosmetic; the readings under it are not.  A timeout is reported
as a plain failure, so the only evidence of *why* a unit stopped is what it was
spending when the deadline passed -- and in CI, where nobody can attach to the
process, it is the only evidence there will ever be.
"""

import io
import subprocess
import sys
import time

from devkit.progress import ConsoleProgress, _Usage


def _sleeper():
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])


def test_usage_keeps_the_peak_of_what_it_saw():
    process = _sleeper()
    try:
        usage = _Usage()
        usage.attach(process.pid)
        rss = usage.sample()
        assert rss > 0
        assert usage.peak >= rss
    finally:
        process.kill()


def test_usage_snapshot_reports_cpu_and_memory():
    process = _sleeper()
    try:
        usage = _Usage()
        usage.attach(process.pid)
        cpu, rss = usage.snapshot()
        assert cpu >= 0.0
        assert rss > 0
    finally:
        process.kill()


def test_usage_reports_nothing_for_a_process_that_is_gone():
    """A dead process has no reading, and None is how the note learns to say so."""
    process = _sleeper()
    process.kill()
    process.wait()
    usage = _Usage()
    usage.attach(process.pid)
    assert usage.sample() == 0
    assert usage.snapshot() is None


def test_console_progress_samples_off_a_terminal(monkeypatch):
    """CI is off a terminal, and that is where the peak is the only evidence.

    Sampling happens in the rich column, which rich's refresh thread evaluates
    whether or not it has a terminal to draw on -- so there is no separate
    polling loop here. That is worth one test rather than a comment: if rich
    ever stops evaluating columns it cannot draw, the peak silently returns to
    zero, which is the bug this replaced.
    """
    monkeypatch.setattr(sys, "__stderr__", io.StringIO())
    process = _sleeper()
    progress = ConsoleProgress()
    try:
        progress.start("compiling something")
        progress.attach(process.pid)
        deadline = time.monotonic() + 5.0
        while progress.peak_rss == 0 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert progress.peak_rss > 0
    finally:
        progress.finish("compiling something", None)
        process.kill()
    assert "compiling something ..." in sys.__stderr__.getvalue()
