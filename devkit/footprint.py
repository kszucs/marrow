"""What the AOT lane costs in machine code.

`marrow.expr`'s comptime lane exists to produce small binaries, and the only way
to know it still does is to build the same programs and measure them.  Each
program under `benchmarks/binary_size/` is one gate; this module builds them,
measures them, and holds them to a recorded floor.

Two things are **derived from the tree rather than listed here**, because both
lists went stale before: the gates are whichever programs exist in the gate
directory, and the module buckets come from marrow's own source layout.  The
previous hand-written bucket list named four `marrow::expr::*` modules that had
been renamed, so they silently reported 0 for every gate -- a table that looks
complete and measures nothing.
"""

import functools
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from rich import box
from rich.console import Console
from rich.table import Table

from .mojo import BuildOptions


class MachO:
    """One built binary, and what the platform tools say about it.

    Parsing is separated from running so the fiddly part -- `nm` output whose
    symbol names contain spaces -- is testable without a compiler.
    """

    def __init__(self, path, runner):
        self.path = Path(path)
        self._runner = runner
        self._output = {}

    def __repr__(self):
        return f"MachO({self.path.name!r})"

    # -- parsing ------------------------------------------------------------

    @staticmethod
    def parse_names(output):
        """Symbol names out of `nm` output.

        A line is `<address> <type> <name...>` when defined and `<blank> U
        <name...>` when not.  Demangled Mojo names routinely contain spaces --
        nested generic signatures -- so the name is everything after the first
        one or two fields, never the last whitespace-separated token.
        """
        names = []
        for line in output.splitlines():
            if not line.strip():
                continue
            fields = line.split(None, 2)
            if len(fields) == 3:
                names.append(fields[2])
            elif len(fields) == 2:
                names.append(fields[1])
        return names

    @staticmethod
    def parse_text(output):
        """Bytes of machine code -- the `__text` *section* -- out of `size -m`.

        Not the `__TEXT` segment and not the file size, both of which are padded
        to a page boundary: 16 KB on Apple Silicon.  Measured 2026-07-29, a
        change adding 1,728 bytes of code moved the stripped file size by 16,504
        and the segment by exactly 16,384, while the symbol count went *down* by
        one.  A gate reading either cannot see a change smaller than a page.
        """
        for line in output.splitlines():
            fields = line.split()
            if len(fields) >= 3 and fields[:2] == ["Section", "__text:"]:
                return int(fields[2])
        return None

    # -- the tools ----------------------------------------------------------

    def _read(self, *argv):
        """Run a tool over this binary, once.

        `names`, `symbols` and `text` are read several times per gate -- the
        size table and the module attribution both want the symbol list -- and
        each is a process. The binary does not change under us, so the output
        is kept.
        """
        if argv not in self._output:
            result = self._runner.run([*argv, self.path], f"{argv[0]} {self.path.name}")
            if not result.ok:
                raise RuntimeError(f"{argv[0]} failed on {self.path}:\n{result.output}")
            self._output[argv] = result.stdout
        return self._output[argv]

    @property
    def names(self):
        return self.parse_names(self._read("nm"))

    @property
    def symbols(self):
        return len(self.names)

    @property
    def text(self):
        return self.parse_text(self._read("size", "-m"))

    @property
    def bytes_on_disk(self):
        return self.path.stat().st_size

    def strip_to(self, path):
        """Copy to *path* and strip it, returning the stripped binary."""
        shutil.copy2(self.path, path)
        stripped = MachO(path, self._runner)
        stripped._read("strip")
        return stripped


class Modules:
    """marrow's module namespaces, read off the source tree.

    A mangled symbol spells its module the way the tree spells it --
    `marrow/kernels/filter.mojo` becomes `marrow::kernels::filter` -- so the
    bucket list is a fact about the checkout, not something to maintain by hand.

    Buckets are **proportional, not a partition**: a mangled name embeds nested
    generic parameters, so one symbol may name several modules and count for
    each.  That is deliberate -- it measures where a binary's code came from.
    """

    SEPARATOR = "::"
    #: Directories of tests and benchmarks; never linked into a gate.
    EXCLUDED = frozenset({"tests"})

    def __init__(self, package_dir):
        self.package_dir = Path(package_dir)

    def names(self):
        """Every module in the package, as a mangled namespace."""
        root = self.package_dir
        if not root.is_dir():
            # Answering [] would drop every row and print an *empty*
            # attribution table -- which reads as "this binary links nothing"
            # rather than as a broken path.
            raise RuntimeError(f"no package source tree at {root}")
        found = []
        for path in sorted(root.rglob("*.mojo")):
            parts = path.relative_to(root).with_suffix("").parts
            if self.EXCLUDED.intersection(parts) or parts[-1] == "__init__":
                continue
            found.append(self.SEPARATOR.join((root.name, *parts)))
        return found

    def count(self, symbols):
        """`{module: symbols naming it}`, keeping only modules that appear.

        Matching includes the trailing separator, because module names nest:
        without it `marrow::kernels::cast` also counts every
        `marrow::kernels::cast_decimal` symbol, and the table prints the two as
        independent rows.  Measured on `query_dynvalue`, that over-reported
        `cast` by 49 of 693 symbols -- 7.6%.

        Deliberately the naive scan.  Pulling namespace runs out of each symbol
        and looking their prefixes up sounds cheaper and is not: mangled names
        are not clean `::`-joined runs, so the inversion both miscounts (filter
        119 -> 25 on `query_dynvalue`) and measures 4x *slower*, because the
        per-symbol split and set work costs more than the substring searches it
        saves.  Whole-sweep cost either way is ~1% of fifteen -O3 Mojo builds.

        Dropping the empty rows keeps the table legible now that the buckets
        are complete rather than curated.
        """
        counts = Counter()
        for module in self.names():
            needle = module + self.SEPARATOR
            hits = sum(1 for symbol in symbols if needle in symbol)
            if hits:
                counts[module] = hits
        return counts


@dataclass(frozen=True)
class Measurement:
    """One gate program, built and measured."""

    name: str
    unstripped: int
    stripped: int
    symbols: int
    symbols_stripped: int
    text: int


class Gates:
    """The gate programs: build them, measure them, attribute their symbols."""

    #: The gate every other one is reported as a multiple of.  A policy choice,
    #: so it is named here; which gates *exist* is not, so they are discovered.
    BASELINE = "query_streaming"

    def __init__(self, repo, toolchain, runner):
        self.directory = repo.footprint_dir
        self._toolchain = toolchain
        self._runner = runner
        self.modules = Modules(repo.package_dir)

    def available(self):
        """Whichever programs are in the gate directory, in report order."""
        return sorted(path.stem for path in self.directory.glob("*.mojo"))

    def resolve(self, wanted):
        """Narrow to *wanted*, always keeping the ratio baseline."""
        available = self.available()
        if not wanted:
            return available
        unknown = set(wanted) - set(available)
        if unknown:
            raise ValueError(f"unknown gate(s): {', '.join(sorted(unknown))}")
        return [
            name for name in available if name in set(wanted) or name == self.BASELINE
        ]

    def source(self, name):
        return self.directory / f"{name}.mojo"

    @functools.cache
    def binary(self, name):
        """One `MachO` per artefact, so its tool output is read once.

        `measure` wants the symbol *count* and `attribute` the symbol *names*;
        minting a fresh object for each ran `nm` twice per gate and parsed the
        same output twice, which made `MachO`'s own memo unreachable.
        """
        return MachO(self.directory / name, self._runner)

    @functools.cache
    def stripped(self, name):
        return MachO(self.directory / f"{name}_stripped", self._runner)

    def build(self, name):
        """Build one gate at -O3 and leave a stripped copy beside it."""
        # Anything read from the previous artefact is stale once it is gone.
        self.binary.cache_clear()
        self.stripped.cache_clear()
        binary, stripped = self.binary(name), self.stripped(name)
        # Remove the previous run's artifacts first: `mojo build` leaves them in
        # place when it fails, so measuring without this reports a stale
        # binary's size as if the failed build had succeeded.
        binary.path.unlink(missing_ok=True)
        stripped.path.unlink(missing_ok=True)
        result = self._toolchain.build(
            self.source(name),
            binary.path,
            BuildOptions.for_size_gate(),
            f"building {name}",
        )
        if not result.ok:
            raise RuntimeError(result.output)
        binary.strip_to(stripped.path)

    def build_all(self, names):
        """Build every gate; return the names that failed.

        One broken gate must not blind the rest of the sweep -- it used to abort
        on the first failure, so a single program left behind by an API change
        hid the numbers for every gate after it.  Callers still fail overall.
        """
        failed = []
        for name in names:
            try:
                self.build(name)
            except RuntimeError:
                failed.append(name)
        return failed

    def measure(self, name):
        binary, stripped = self.binary(name), self.stripped(name)
        return Measurement(
            name=name,
            unstripped=binary.bytes_on_disk,
            stripped=stripped.bytes_on_disk,
            symbols=binary.symbols,
            symbols_stripped=stripped.symbols,
            text=stripped.text,
        )

    def text_size(self, name):
        """Just the `__text` figure, which is all the recorded floor compares."""
        return self.stripped(name).text

    def attribute(self, names):
        """`{gate: {module: symbol count}}` for the per-module breakdown."""
        return {name: self.modules.count(self.binary(name).names) for name in names}


class Baseline:
    """The recorded `__text` floor each gate must stay under."""

    def __init__(self, path):
        self.path = Path(path)
        self._data = json.loads(self.path.read_text())

    @property
    def threshold_pct(self):
        return self._data["threshold_pct"]

    @property
    def gates(self):
        return self._data["gates"]

    def check(self, measured):
        """`[(name, floor, measured, delta, pct, regressed)]`, in record order."""
        rows = []
        for name, floor in self.gates.items():
            text = measured.get(name)
            if text is None:
                # `size -m` printed no `__text` line, or the gate was never
                # measured.  Either way the floor cannot be compared, and
                # `None - floor` would surface as a TypeError three frames away.
                raise RuntimeError(f"no __text measurement for gate {name!r}")
            if not floor:
                raise RuntimeError(f"gate {name!r} has a zero baseline")
            delta = text - floor
            pct = 100.0 * delta / floor
            rows.append((name, floor, text, delta, pct, pct > self.threshold_pct))
        return rows

    def update(self, measured):
        """Re-record the floor, keeping the `_comment` changelog intact."""
        self._data["gates"] = dict(measured)
        self.path.write_text(json.dumps(self._data, indent=2) + "\n")


class Report:
    """Renders what the gates measured."""

    def __init__(self):
        self.console = Console(highlight=False)

    def _table(self, title, columns, rows):
        table = Table(title=title, box=box.SIMPLE_HEAD)
        table.add_column(columns[0], no_wrap=True)
        for column in columns[1:]:
            table.add_column(column, justify="right", no_wrap=True)
        for row in rows:
            table.add_row(*row)
        return table

    def comparison(self, measurements, failed, attribution):
        self.console.print(
            self._table(
                "Binary size",
                ("binary", "unstripped", "stripped", "syms", "syms(strip)", "__text"),
                [
                    (
                        m.name,
                        f"{m.unstripped:,}",
                        f"{m.stripped:,}",
                        f"{m.symbols:,}",
                        f"{m.symbols_stripped:,}",
                        f"{m.text:,}",
                    )
                    for m in measurements
                ]
                + [
                    (name, "[red]BUILD FAILED[/red]", "", "", "", "") for name in failed
                ],
            )
        )

        baseline = next((m for m in measurements if m.name == Gates.BASELINE), None)
        if baseline is not None:
            self.console.print(
                self._table(
                    f"__text ratio vs. {baseline.name} (code only, not page-padded)",
                    ("binary", "ratio"),
                    [(m.name, f"{m.text / baseline.text:.1f}x") for m in measurements],
                )
            )
        self.console.print(
            "Compare runs on __text. The stripped column is page-granular (16 KB "
            "on Apple Silicon) and moves in steps -- do not quote deltas from it.",
            style="dim",
        )
        self.console.print(self._attribution(attribution))

    @staticmethod
    def _attribution(attribution):
        """One row per module any measured gate links, zero rows omitted."""
        modules = sorted(
            {module for counts in attribution.values() for module in counts}
        )
        table = Table(title="Symbols by module (unstripped)", box=box.SIMPLE_HEAD)
        table.add_column("module", no_wrap=True)
        for name in attribution:
            table.add_column(name, justify="right", no_wrap=True)
        for module in modules:
            table.add_row(
                module,
                *[f"{counts.get(module, 0):,}" for counts in attribution.values()],
            )
        return table

    def gate(self, rows, threshold_pct):
        """Print the comparison against the floor; return the regressed gates."""
        self.console.print(
            self._table(
                "Binary-size gate",
                ("gate", "floor", "measured", "delta", "pct", ""),
                [
                    (
                        name,
                        f"{floor:,}",
                        f"{text:,}",
                        f"{delta:+,}",
                        f"{pct:+.3f}%",
                        "[red]REGRESSION[/red]" if regressed else "",
                    )
                    for name, floor, text, delta, pct, regressed in rows
                ],
            )
        )
        regressed = [row[0] for row in rows if row[5]]
        if regressed:
            self.console.print(
                f"FAIL: {', '.join(regressed)} grew more than {threshold_pct}% in "
                "`__text` versus the recorded floor.",
                style="bold red",
            )
            self.console.print(
                "If the growth is intentional, re-run with --update and commit "
                "the updated baseline."
            )
        else:
            self.console.print(
                f"OK: no gate grew more than {threshold_pct}%.", style="green"
            )
        return regressed
