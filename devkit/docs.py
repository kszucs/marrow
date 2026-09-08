"""Compile every Mojo listing in the docs, so the guides cannot rot.

Quarto executes the Python cells at render time, which keeps that half honest.
Mojo has no such engine, and this stands in for it.  Two kinds of listing are
covered:

1. **Extracted programs** -- the files under the snippets directory, which a
   page pulls in with `{{< include >}}`, so the rendered listing and the
   compiled file are the same bytes.
2. **Inline blocks** -- ```` ```mojo ```` fences written directly in a page.
   These are compiled too, by lifting the `from ... import ...` lines to module
   scope and wrapping the rest in a `main()`.

A block that is deliberately partial -- a signature, a chained call with no
surrounding program -- opts out with ```` ```{.mojo .fragment} ````.  Opting out
is a claim that the block is illustrative, not runnable; prefer moving a real
example into the snippets directory over marking it a fragment.

**Judged by its output, not by its exit status alone.**  `mojo build` reports a
parse failure and still exits 0 -- see CLAUDE.md.  Grepping for `error:` is
therefore necessary but *not* sufficient: a compiler crash prints `Stack dump:`
and no `error:` at all, and a link failure need not print one either, so a build
that died would be reported as ok.  `MojoToolchain.reports_errors` reads the
status *and* the output, which is why this goes through the toolchain rather
than calling `subprocess` itself -- and it inherits the timeout with it, which a
CI job compiling ~22 programs with no deadline otherwise lacks.
"""

import re
import tempfile
from pathlib import Path

from .mojo import BuildOptions

#: ```mojo / ```{.mojo ...} -- captures the attribute string and the body.
FENCE = re.compile(r"^```(?:\{\.mojo([^}]*)\}|mojo)\s*\n(.*?)^```", re.M | re.S)


def as_program(body):
    """Lift imports to module scope and wrap the remainder in `main()`.

    An inline block is written to read well on the page, so it is a fragment of
    a program rather than one.  A block that already declares `main` is taken
    as written.
    """
    if re.search(r"^def main\(", body, re.M):
        return body
    imports, rest = [], []
    for line in body.splitlines():
        (imports if re.match(r"^\s*(from|import)\s", line) else rest).append(line)
    while rest and not rest[0].strip():
        rest.pop(0)
    indented = "\n".join(f"    {line}" if line.strip() else "" for line in rest)
    return "\n".join(imports) + "\n\n\ndef main() raises:\n" + indented + "\n"


class SnippetCheck:
    """Every Mojo listing the docs publish, compiled one at a time."""

    #: A listing is a few hundred lines at most; anything past this is a hang.
    TIMEOUT = 600

    def __init__(self, repo, toolchain):
        self._repo = repo
        self._mojo = toolchain

    def listings(self):
        """`([(label, source)], fragments)` -- everything to compile, in a
        stable order, and how many blocks opted out.

        The opt-outs are counted here rather than left implicit because the
        number going up is the guides drifting out of reach, and a caller that
        only sees what compiled cannot tell.
        """
        found = [
            (str(src.relative_to(self._repo.root)), src.read_text())
            for src in sorted(self._repo.snippets_dir.glob("*.mojo"))
        ]
        fragments = 0
        for page in sorted(self._repo.docs_dir.rglob("*.qmd")):
            for i, (attrs, body) in enumerate(FENCE.findall(page.read_text())):
                if ".fragment" in (attrs or ""):
                    fragments += 1
                elif "{{<" not in body:
                    # An `include` names a file already collected above, so
                    # compiling the fence too would just duplicate it.
                    label = f"{page.relative_to(self._repo.docs_dir)}#{i}"
                    found.append((label, as_program(body)))
        return found, fragments

    def compile(self, label, source, workdir):
        """Build one listing; return its output if it failed, else `None`."""
        src = workdir / f"{label}.mojo"
        src.write_text(source)
        result = self._mojo.build(
            src, workdir / label, BuildOptions.for_docs(), f"building {label}"
        )
        return result.output if self._mojo.reports_errors(result) else None

    def run(self, report):
        """Compile everything and report as it goes; True when all built."""
        listings, fragments = self.listings()
        failed = []
        with tempfile.TemporaryDirectory() as tmp:
            for n, (label, source) in enumerate(listings):
                log = self.compile(f"snippet_{n}", source, Path(tmp))
                if log is None:
                    report.ok(label)
                else:
                    failed.append(label)
                    report.failure(label, log)
        report.summary(len(listings), failed, fragments)
        return not failed


class Report:
    """What the check prints.  Separate so a test can assert on the result
    rather than on captured stdout."""

    def ok(self, label):
        print(f"ok   {label}")

    def failure(self, label, log):
        diagnostics = [line for line in log.splitlines() if "error:" in line]
        print(f"FAIL {label} -- {len(diagnostics) or 'no'} error(s)")
        # A crash or a link failure carries no `error:` line at all, so fall
        # back to the whole log rather than printing nothing.
        print("\n".join(f"     {ln}" for ln in (diagnostics or log.splitlines())))

    def summary(self, total, failed, fragments):
        print(
            f"\n{total - len(failed)}/{total} Mojo listings compiled"
            f" ({fragments} marked .fragment)"
        )
