"""Regenerate fixtures and expectations.

    pixi run -e bench python golden/regenerate.py

Expectations come from **DuckDB**, never from marrow: an expectation captured
from the engine under test enshrines whatever that engine currently does, which
is how a golden corpus quietly becomes a record of its own bugs.

A case's SQL is the **first paragraph** of its docstring; a blank line ends
it and everything after is commentary.

Output is committed as text so that a change to an expectation shows up as a
reviewable diff. `conftest.py` parses it back and materialises the per-case
Arrow IPC files the Mojo lane reads.
"""

import ast
import sys
from pathlib import Path

import duckdb
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).parent))
import fixtures  # noqa: E402
from expfmt import render  # noqa: E402

DIR = Path(__file__).parent


def cases(module_path):
    """Every `def test_*` in a golden module, with its docstring SQL."""
    tree = ast.parse(module_path.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            doc = ast.get_docstring(node)
            if not doc:
                raise SystemExit(f"golden: {node.name} has no SQL docstring")
            # SQL is the docstring's **first paragraph**; anything after a
            # blank line is commentary for a human. Without the split, prose
            # explaining a case is concatenated onto its query and DuckDB
            # fails to parse it.
            sql = doc.split("\n\n", 1)[0]
            yield node.name, " ".join(sql.split())


def main():
    written = fixtures.write_all()
    print(f"fixtures: {', '.join(written)}")

    con = duckdb.connect()
    for name in written:
        con.register(name, fixtures.read(name))

    for module_path in sorted(DIR.glob("test_*.py")):
        blocks = []
        for case, sql in cases(module_path):
            table = pa.table(con.execute(sql).arrow())
            blocks.append(f"== {case}\n{render(table)}")
            print(f"  {case}: {table.num_rows} rows")
        out = module_path.with_suffix(".exp")
        out.write_text("\n\n".join(blocks) + "\n")
        print(f"{out.name}: {len(blocks)} cases")


if __name__ == "__main__":
    main()
