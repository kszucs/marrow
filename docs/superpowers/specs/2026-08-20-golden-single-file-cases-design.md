# Golden corpus: one file per case

*2026-08-20*

## Problem

A golden case is currently spread across three files. `test_kleene.py` holds
the SQL (in a docstring) and the runtime-lane expression, `test_kleene.mojo`
holds the AOT-lane expression, and `test_kleene.exp` holds the expectation for
both. Reading one case means reading three files and matching by name, and
nothing keeps the two lanes' expressions describing the same query — they
drifted once already, when the AOT lane had no boolean column leaf and the
Python lane silently tested a different thing.

Worse, the divergence is invisible. Marrow's stated design target is that a
Mojo expression and a Python expression are *the same expression* — the Mojo
one compiled ahead of time, the Python one evaluated through the runtime lane
and the bindings. Today all 69 cases are spelled differently in the two lanes,
and nothing measures that or pushes it toward zero.

## Design

**One case is one file: `golden/cases/<name>.mojo`.** It is Mojo source, and
Mojo is the source of truth. Python is derived from it by a mechanical
transpile. That direction is not arbitrary — Mojo is the constrained lane (no
`**kwargs`, a dtype required at comptime), so its spelling is the intersection
of the two grammars and the one Python can always accept.

### Case file anatomy

```mojo
def test_golden_kleene_column_and() raises:
    """SELECT p, q, p AND q AS r FROM flags

    Reads bool *columns*, not derived predicates — the shape the AOT lane
    could not express until `col` gained a `BoolType` overload.

    -- expected
    p:bool<TAB>q:bool<TAB>r:bool
    True<TAB>True<TAB>True
    True<TAB>False<TAB>False
    True<TAB>NULL<TAB>NULL
    """
    var t = table("flags")
    check(
        t.project(
            ["p", "q", "r"],
            [col("p", bool_), col("q", bool_), col("p", bool_) & col("q", bool_)],
        )
    )
```

(`<TAB>` above stands for a literal tab; the expected block is the typed TSV
`expfmt.py` already renders, `name:type` per column and `NULL` for a null.)

The docstring carries three things in a fixed order: the SQL as the first
paragraph (the existing convention), optional prose for a human, and the
expected table after a `-- expected` marker. The marker is spelled as a SQL
comment because the block above it is SQL.

**A case file carries no import lines.** Each lane's generator emits a fixed
header covering the whole allowed vocabulary — `table`, `check`, `col`, `lit`,
`bool_`, `int64`, `CaseWhen`, and so on — and the Python runner execs case
bodies in a namespace holding exactly those same names. That set is the
convergence contract, written down once per lane rather than scattered across
69 files. The cost is that a case file is not standalone-compilable, so an LSP
flags unresolved names; it is never compiled standalone anyway, and
`mojo format` is purely syntactic so formatting still works.

**`check(plan)` takes no case name.** Mojo cannot introspect its own function
name, so both generators inject it, rewriting `check(` to
`check("<case name>", `. One rule applied identically in both lanes keeps the
case text lane-neutral.

### The two generators

**Mojo.** `golden/conftest.py`, on import, concatenates every `cases/*.mojo`
under the fixed import header into `golden/test_cases.mojo` (generated,
gitignored). One module, so the driver's compile cost stays comparable to
today's. The existing harness then collects it unchanged: `pytest_collect_file`
(`conftest.py:819`) picks up `test_*.mojo` and regex-scans for `def test_*(`
(`conftest.py:24`). No new Mojo machinery exists — in particular there is no
hand-written `test_cases.mojo`, because Mojo has no `eval` and a hand-written
one could not parse anything.

**Python.** `golden/test_cases.py`, on import, transpiles each case body and
injects the resulting function into module globals. Item names then match the
Mojo lane exactly, so `-k` selects symmetrically across both. The transpiler is
three rules:

| Mojo | Python |
|---|---|
| `def f() raises:` | `def f():` |
| `var x = …` | `x = …` |
| everything else | verbatim |

This holds only while case bodies stay inside the intersection of the two
grammars, which costs one constraint: **no type annotations in a case body**
(`var t: DynRelation = …` would require `DynRelation` in the Python namespace).
Everything the corpus uses today — `~`, `&`, `|`, list literals, keyword
arguments, multi-line parenthesised expressions, `True`/`False`/`None` — is
already common to both.

### Convergence rule

**Python grows the Mojo-shaped API, never the reverse.** Mojo cannot take
`**kwargs` and needs a dtype at comptime, so `project(names, values)` and
`col(name, dtype)` are the shapes both lanes must speak. Every convergence gap
found during migration is closed by an addition on the Python side.

The prototype needs exactly two: `col(name, dtype=None)`
(`python/marrow/_expr_column.py:415`, currently `col(name)`) and a positional
`project(names, values)` form alongside the keyword one
(`python/marrow/expr.py:182`). Dtype names such as `bool_` and `int64` are
already exported from `marrow`.

### Expectation pipeline

`golden/casefmt.py` is new and owns the format end to end: parse
`(name, sql, prose, expected)` out of a case file, render a typed table back
into a docstring. `expfmt.py`'s typed-TSV render and parse survive inside it;
its `== name` block splitter is deleted, since one file is now one case.

`regenerate.py` (`pixi run -e bench python golden/regenerate.py`) parses every
`cases/*.mojo`, runs the SQL through DuckDB against the fixtures, and rewrites
each docstring in place, replacing everything after `-- expected` and appending
the marker when absent. Expectations still come from DuckDB and never from
marrow — an expectation captured from the engine under test enshrines that
engine's current bugs. They stay committed as text, so a changed expectation is
a reviewable diff.

`golden/conftest.py` derives two artefacts from that committed text on import,
so neither can drift from what a reviewer saw: `.exp/<name>.arrow` per case
(typed Arrow IPC, which is what the Mojo lane reads) and
`golden/test_cases.mojo`.

`golden/helpers.py` is new and holds the Python namespace. The `Golden` fixture
disappears, so the `--morsel-size` and `--num-threads` options it read off
`request.config` move to module state set in `pytest_configure`. That keeps the
signature `table(name)` identical in both lanes instead of leaking a config
argument into every case. On the Mojo side, `helpers.mojo`'s `fixture()`
becomes `table()` and wraps `in_memory_table` itself.

### Unconvergeable cases

A case whose two lanes genuinely cannot converge yet would block its own
migration, and the old files it could otherwise hide in are being deleted. To
keep that from silently reintroducing two spellings, a case file may declare
`-- skip python` or `-- skip mojo` on a line of its own in its docstring,
after the SQL paragraph and before `-- expected`. The runner honours it and
migration reports the list, so divergence stays visible and countable rather
than dissolving back into prose.

## Phases

Each phase ends at a review gate.

### Phase 1 — prototype

`kleene_column_and` only, alongside the untouched existing corpus. It exercises
bool columns, `project`, and Kleene null semantics, and needs only the two
Python additions above.

Deliverables: `casefmt.py` parse and render, `helpers.py` namespace, minimal
`conftest.py` codegen for one case, `col(name, dtype=None)`, positional
`project(names, values)`.

Four things are verified rather than assumed:

1. `mojo format` preserves tabs and trailing whitespace inside a docstring. If
   it does not, the expected block becomes pipe-delimited (` | `) instead of
   tab-separated, and string values containing a pipe are escaped.
2. `golden/conftest.py` is imported before `pytest_collect_file` reaches
   `golden/test_cases.mojo`, so the generated file exists in time. If not,
   generation moves to a `pytest_configure` hook in the root `conftest.py`.
3. Both lanes pass:
   `pixi run -e dev pytest golden/test_cases.mojo golden/test_cases.py -k kleene_column_and`
4. The generated module's compile cost is measured against today's
   `golden/test_kleene.mojo`, giving phase 3 a baseline.

### Phase 2 — machinery

`regenerate.py` rewritten to parse case files and rewrite docstrings in place.
Full transpiler with errors that name the case file and line. The harness gets
its own tests, as `conftest.py` already carries `test_write_driver_*`: a
`casefmt` round trip and the three transpile rules.

### Phase 3 — migration

All 69 cases ported; the `test_*.mojo`, `test_*.py` and `test_*.exp`
triples and `expfmt.py` deleted.

Known convergence gaps, each a Python-side addition under the standing rule:

| area | Mojo | Python today |
|---|---|---|
| join | `how=JOIN_INNER, strictness=JOIN_ALL` | `join_type="inner"` |
| sort | `sort([col("eid", int64)], [True])` | `sort_by(...)` |
| conditional | `CaseWhen` / `Coalesce` / `FillNull` | `if_else` / `.coalesce()` / `.fill_null()` |
| aggregate | positional | `aggregate(by=, **named)` |

## Success criteria

- Every golden case is one file under `golden/cases/`.
- Both lanes run from that file, the Mojo one AOT-compiled, the Python one
  through the runtime lane and the bindings.
- Expectations still come from DuckDB and are still committed as reviewable
  text.
- `pixi run -e bench python golden/regenerate.py` produces an empty diff on an
  unmodified corpus.
- The count of cases whose two lanes are byte-identical is reported. It is 0
  of 69 today; the target is all of them, and anything short of that is named
  by a `-- skip` marker rather than hidden.

## Risks

**Formatter interference.** `mojo format` could reflow or strip whitespace in
the expected block. Checked at the phase 1 gate; the pipe-delimited fallback
is the mitigation.

**Compile cost.** 69 concatenated cases in one module is more instantiations
than any single golden file today. Measured at the phase 1 gate and again in
phase 3; if it regresses badly the generator can emit several modules instead
of one, since the harness collects any `test_*.mojo`.

**Transpiler scope creep.** Three rules only. A case needing a fourth rule is
a signal that the case has left the shared grammar — fix the case, or the API,
not the transpiler.
