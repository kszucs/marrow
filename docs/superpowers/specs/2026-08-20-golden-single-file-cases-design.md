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
from golden.helpers import table
from marrow.dtypes import bool_
from marrow.expr.builders import col
from marrow.expr.relations import DynRelation


def plan() raises -> DynRelation:
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
    var q = t.project(
        ["p", "q", "r"],
        [col("p", bool_), col("q", bool_), col("p", bool_) & col("q", bool_)],
    )
    return q
```

(`<TAB>` above stands for a literal tab; the expected block is the typed TSV
`expfmt.py` already renders, `name:type` per column and `NULL` for a null.
**String cells are quoted** — `'  pad  '`, `''` — see the formatter finding
under Phases below.)

The docstring carries three things in a fixed order: the SQL as the first
paragraph (the existing convention), optional prose for a human, and the
expected table after a `-- expected` marker. The marker is spelled as a SQL
comment because the block above it is SQL.

**A case file is a standalone Mojo module, and its name is the file name.**
The Mojo lane *imports* each case rather than copying its body, so what
compiles is the file you edit and a query exists in exactly one place. That
costs a wrapper — Mojo has no top-level statements in a package module, so a
bare expression cannot be a file — but the wrapper's name is the fixed word
`plan`, never the case's own, and nothing inside the file repeats its identity.
Each case therefore carries its own imports, three to six lines; a shared
header is impossible once the modules compile separately, and
`from ... import *` is banned repo-wide.

**A case returns its plan; it never calls `check`.** `check` needs the case
name, and Mojo cannot introspect a function's own name — so the generated
wrapper supplies it from the file name. This is what keeps the case body free
of its own identity, and it removes `check` from the case vocabulary
altogether.

### The two generators

**Mojo.** `runner.py`, on import from `conftest.py`, writes
`golden/test_cases.mojo` (generated, gitignored): one import and one
three-line wrapper per case, no bodies. The harness then collects it unchanged
— `pytest_collect_file` (`conftest.py:819`) picks up `test_*.mojo` and
regex-scans for `def test_*(` (`conftest.py:24`), so the wrappers must exist;
what they must not do is restate the query. Measured: 69 separately-compiled
case modules cost 161s cold, against 160s for a single concatenated module, so
module count is not the lever it was feared to be.

**Python.** `runner.py` transpiles each case body and injects the resulting
function into `test_cases.py`'s module globals. Item names then match the
Mojo lane exactly, so `-k` selects symmetrically across both. The transpiler is
three rules:

| Mojo | Python |
|---|---|
| `def f() raises:` | `def f():` |
| `var x = …` | `x = …` |
| everything else | verbatim |

This holds only while case bodies stay inside the intersection of the two
grammars, which costs two constraints:

- **No type annotations.** `var t: DynRelation = …` would require
  `DynRelation` in the Python namespace.
- **No transfer sigils.** `q^` is Mojo-only syntax with no Python reading, so
  every helper a case body calls must take its arguments borrowed.

Everything the corpus uses today — `~`, `&`, `|`, list literals, keyword
arguments, multi-line parenthesised expressions, `True`/`False`/`None` — is
already common to both.

### Convergence rule

**Python grows the Mojo-shaped API, never the reverse.** Mojo cannot take
`**kwargs` and needs a dtype at comptime, so `project(names, values)` and
`col(name, dtype)` are the shapes both lanes must speak. Every convergence gap
found during migration is closed by an addition on the Python side.

Two additions are genuinely marrow's and were made there:
`col(name, dtype=None)` (`python/marrow/_expr_column.py`) and a positional
`project(names, values)` / `with_columns(names, values)` form alongside the
keyword one (`python/marrow/expr.py`).

**The rest are golden-local shims, and deliberately so.** Reading the whole
corpus showed the gap is far wider than this spec first assumed, and much of
it is the fused lane's *internal* vocabulary —
`AggExpr.of[NumericAgg[SumKernel, Int64Type]](x)`, `NumericCast[Float64Type]`,
`List[BoxedValue]()`, `Upper(x)`. Growing marrow's public Python API to speak
that would be a bad trade: nobody should write `AggExpr.of[NumericAgg[...]]`
in Python when `.sum()` exists. So the bridge lives in `helpers.py`, where it
is countable, and `helpers.SHIMS` is the metric — 48 entries today, target
zero, guarded by `test_golden_shims_are_declared` so it cannot go stale in
either direction.

### Module layout

Five files, three of which carry content:

| file | role |
|---|---|
| `golden/helpers.mojo` | the Mojo vocabulary — `table`, `check`, `values_equal` |
| `golden/helpers.py` | the Python vocabulary, mirroring it, plus the `NAMESPACE` dict case bodies exec in |
| `golden/runner.py` | all machinery — fixture definitions, case parse and render, DuckDB regeneration, `.exp` and `test_cases.mojo` codegen, the transpiler |
| `golden/conftest.py` | pytest hook shim — options, config, triggers codegen on import |
| `golden/test_cases.py` | collection shim — `runner.install(helpers.NAMESPACE, globals())` |

`expfmt.py`, `fixtures.py` and `regenerate.py` are deleted and folded into
`runner.py`; `casefmt.py` is never created. The two shims exist only because
pytest's naming rules force them — hooks must live in a file called
`conftest.py`, and collected functions must live in one matching `test_*.py` —
and each is a handful of lines.

Dependencies run one way: `runner.py` imports nothing from `golden`,
`helpers.py` imports `runner`, and the two shims import both. `runner.py` must
not import `helpers.py`; that would close a cycle, since the transpiler lives
in `runner` and the namespace it execs against lives in `helpers`.

**`runner.py` must not import duckdb at module scope.** The split between a
format module and a regeneration script is *why* `expfmt.py` exists today: the
`dev` environment has no duckdb, and comparing against an expectation does not
need one. Folding them back together is only safe with `import duckdb` inside
the regeneration function. Regeneration becomes
`pixi run -e bench python golden/runner.py`, replacing `golden/regenerate.py`.

### Expectation pipeline

`runner.py` owns the case format end to end: parse `(name, sql, prose,
expected)` out of a case file, render a typed table back into a docstring.
`expfmt.py`'s typed-TSV render and parse survive inside it; its `== name` block
splitter is deleted, since one file is now one case.

Run as a script, it parses every `cases/*.mojo`, runs each SQL through DuckDB
against the fixtures, and rewrites the docstring in place — replacing
everything after `-- expected`, appending the marker when absent. Expectations
still come from DuckDB and never from marrow, since an expectation captured
from the engine under test enshrines that engine's current bugs. They stay
committed as text, so a changed expectation is a reviewable diff.

Imported by `conftest.py`, it derives two artefacts from that committed text so
neither can drift from what a reviewer saw: `.exp/<name>.arrow` per case (typed
Arrow IPC, which is what the Mojo lane reads) and `golden/test_cases.mojo`.

The `Golden` fixture disappears, so the `--morsel-size` and `--num-threads`
options it read off `request.config` move to module state in `runner.py`, set
from `conftest.py`'s `pytest_configure`. That keeps the signature `table(name)`
identical in both lanes instead of leaking a config argument into every case.
On the Mojo side, `helpers.mojo`'s `fixture()` becomes `table()` and wraps
`in_memory_table` itself.

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

Deliverables: `runner.py` with case parse, render and single-case codegen;
`helpers.py` namespace; the `conftest.py` and `test_cases.py` shims;
`col(name, dtype=None)`; positional `project(names, values)`.

Four things are verified rather than assumed:

1. `mojo format` preserves tabs and trailing whitespace inside a docstring.
   **Resolved, and half of it was false.** Tabs survive; trailing whitespace
   does not. The `words` fixture holds `"  pad  "`, so the formatter silently
   turned two expectations into different strings and the corpus asserted the
   wrong answer. The pipe-delimited fallback this spec proposed would *not*
   have helped — a trailing cell still ends the line. The fix is to quote
   string cells, so every line ends in a printable character; it also lets
   string data contain a tab or a newline, which the bare format could not
   represent at all. `golden/cases` and `golden/helpers.mojo` then joined the
   `fmt` tasks.
2. `golden/conftest.py` is imported before `pytest_collect_file` reaches
   `golden/test_cases.mojo`, so the generated file exists in time. If not,
   generation moves to a `pytest_configure` hook in the root `conftest.py`.
3. Both lanes pass:
   `pixi run -e dev pytest golden/test_cases.mojo golden/test_cases.py -k kleene_column_and`
4. The generated module's compile cost is measured against today's
   `golden/test_kleene.mojo`, giving phase 3 a baseline.

### Phase 2 — machinery

`runner.py`'s regeneration path rewritten to parse case files and rewrite
docstrings in place, with `import duckdb` confined to it. Full transpiler with
errors that name the case file and line. The harness gets its own tests, as the
root `conftest.py` already carries `test_write_driver_*`: a case-file round trip
and the three transpile rules.

### Phase 3 — migration

All 69 cases ported; the `test_*.mojo`, `test_*.py` and `test_*.exp` triples
deleted, along with `expfmt.py`, `fixtures.py` and `regenerate.py`.

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
- `pixi run -e bench python golden/runner.py` produces an empty diff on an
  unmodified corpus.
- `pixi run -e dev pytest golden/` works with no duckdb installed.
- All 69 cases run in both lanes from one text, so the byte-identical count is
  69 of 69 and no case needs a `-- skip` marker. What remains is measured
  instead by `helpers.SHIMS` — the vocabulary the two lanes still spell
  differently — which stands at 48 and should fall to zero.

## Risks

**Formatter interference.** Confirmed real, and fixed by quoting string cells
rather than by the pipe-delimited fallback this spec originally proposed —
which would not have worked. See phase 1, checkpoint 1.

**Compile cost.** 69 concatenated cases in one module is more instantiations
than any single golden file today. Measured at the phase 1 gate and again in
phase 3; if it regresses badly the generator can emit several modules instead
of one, since the harness collects any `test_*.mojo`.

**Transpiler scope creep.** Three rules only. A case needing a fourth rule is
a signal that the case has left the shared grammar — fix the case, or the API,
not the transpiler.
