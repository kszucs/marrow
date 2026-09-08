#!/usr/bin/env python3
"""Compile every Mojo listing in the docs, so the guides cannot rot.

Quarto executes the Python cells at render time, which keeps that half honest.
Mojo has no such engine, and this stands in for it. Two kinds of listing are
covered:

1. **Extracted programs** -- `docs/snippets/*.mojo`, real files a page pulls in
   with `{{< include >}}`, so the rendered listing and the compiled file are the
   same bytes.
2. **Inline blocks** -- ```` ```mojo ```` fences written directly in a page.
   These are compiled too, by lifting the `from ... import ...` lines to module
   scope and wrapping the rest in a `main()`.

A block that is deliberately partial -- a signature, a chained call with no
surrounding program -- opts out with ```` ```{.mojo .fragment} ````. Opting out
is a claim that the block is illustrative, not runnable; prefer moving a real
example into `docs/snippets/` over marking it a fragment.

**Judged by its output, not by its exit status alone.** `mojo build` reports a
parse failure and still exits 0 -- see CLAUDE.md. Grepping for `error:` is
therefore necessary but *not* sufficient: a compiler crash prints `Stack dump:`
and no `error:` at all, and a link failure need not print one either, so a
build that died would be reported as ok. `MojoToolchain.reports_errors` reads
the status *and* the output, which is why this goes through devkit rather than
calling `subprocess` itself -- and it inherits the timeout with it, which a CI
job compiling ~22 programs with no deadline otherwise lacks.
"""

import re
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from devkit.mojo import (  # noqa: E402 - must follow the path insertion
    BuildOptions,
    MojoToolchain,
    ProcessRunner,
    Repo,
)
from devkit.progress import ConsoleProgress  # noqa: E402

SNIPPETS = REPO / "docs" / "snippets"
DOCS = REPO / "docs"

#: A listing is a few hundred lines at most; anything past this is a hang.
TIMEOUT = 600

# ```mojo / ```{.mojo ...}  -- captures the attribute string and the body.
FENCE = re.compile(r"^```(?:\{\.mojo([^}]*)\}|mojo)\s*\n(.*?)^```", re.M | re.S)


def toolchain() -> MojoToolchain:
    repo = Repo.locate(REPO)
    return MojoToolchain(ProcessRunner(repo.root, ConsoleProgress(), timeout=TIMEOUT))


def compile_source(mojo: MojoToolchain, source: str, label: str, tmp: Path):
    """Build one Mojo source; return `(failed, combined output)`."""
    src = tmp / f"{label}.mojo"
    src.write_text(source)
    result = mojo.build(src, tmp / label, BuildOptions.for_docs(), f"building {label}")
    return mojo.reports_errors(result), result.output


def as_program(body: str) -> str:
    """Lift imports to module scope and wrap the remainder in `main()`."""
    if re.search(r"^def main\(", body, re.M):
        return body
    imports, rest = [], []
    for line in body.splitlines():
        (imports if re.match(r"^\s*(from|import)\s", line) else rest).append(line)
    while rest and not rest[0].strip():
        rest.pop(0)
    indented = "\n".join(f"    {line}" if line.strip() else "" for line in rest)
    return "\n".join(imports) + "\n\n\ndef main() raises:\n" + indented + "\n"


def main() -> int:
    checks: list[tuple[str, str]] = []

    for src in sorted(SNIPPETS.glob("*.mojo")):
        checks.append((str(src.relative_to(REPO)), src.read_text()))

    fragments = 0
    for page in sorted(DOCS.rglob("*.qmd")):
        for i, (attrs, body) in enumerate(FENCE.findall(page.read_text())):
            if ".fragment" in (attrs or ""):
                fragments += 1
                continue
            if "{{<" in body:  # an `include` -- the real file is checked above
                continue
            label = f"{page.relative_to(DOCS)}#{i}"
            checks.append((label, as_program(body)))

    failed = []
    mojo = toolchain()
    with tempfile.TemporaryDirectory() as tmp:
        for n, (label, source) in enumerate(checks):
            broke, log = compile_source(mojo, source, f"snippet_{n}", Path(tmp))
            if broke:
                failed.append(label)
                diagnostics = [ln for ln in log.splitlines() if "error:" in ln]
                print(f"FAIL {label} -- {len(diagnostics) or 'no'} error(s)")
                # A crash or a link failure carries no `error:` line at all, so
                # fall back to the whole log rather than printing nothing.
                print(
                    "\n".join(f"     {ln}" for ln in (diagnostics or log.splitlines()))
                )
            else:
                print(f"ok   {label}")

    print(
        f"\n{len(checks) - len(failed)}/{len(checks)} Mojo listings compiled"
        f" ({fragments} marked .fragment)"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
