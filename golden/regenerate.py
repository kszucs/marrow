"""Regenerate fixtures and expectations.

    pixi run -e bench python golden/regenerate.py

Expectations come from **DuckDB**, never from marrow: an expectation captured
from the engine under test enshrines whatever that engine currently does, which
is how a golden corpus quietly becomes a record of its own bugs.

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
            sql = ast.get_docstring(node)
            if not sql:
                raise SystemExit(f"golden: {node.name} has no SQL docstring")
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
