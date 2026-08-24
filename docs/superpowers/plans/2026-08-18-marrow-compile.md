# `marrow compile` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `marrow compile <file> [-o out]`, a CLI that turns a Mojo file holding a marrow AOT expression into a small standalone binary whose source paths and scalar constants are supplied on the command line at run time.

**Architecture:** A new `param()` builder joins `col()` and `lit()` in `marrow/exprold/builders.mojo`. It returns an expression node structurally identical to the matching `*Literal` node, except the value lives behind an `ArcPointer[ParamCell]` resolved once per batch in `state()` — so the inner SIMD loop is unchanged and a parameter costs nothing per row. `param()` also records a declaration in a module-level registry; `plan.execute_cli()` drains that registry, parses `argv`, binds the cells, executes, and writes output. The Python CLI is a thin build front-end over `mojo build -O3 -g0 -I <marrow-source>` plus `strip`, with a `--bundle` mode that copies the transitive dylib closure and rewrites the rpath.

**Tech Stack:** Mojo 1.1.0.dev2026081705, pixi, pytest via the repo's `conftest.py` harness, hatchling for the wheel.

**Spec:** `docs/superpowers/specs/2026-08-18-marrow-compile-design.md` — read it before starting. It carries the measured evidence behind every decision here, including why static linking is impossible and why discovery uses a registry rather than a `parameters()` trait method.

## Global Constraints

Copied verbatim from `CLAUDE.md` and the spec. **Every task's requirements implicitly include this section.**

- **Always use `def`, never `fn`.** `fn` is deprecated.
- **Never use `alias` — use `comptime`.** `alias` is deprecated.
- **Relative imports only** for `marrow.*` — `...x` from `marrow/<sub>/tests/`, `..x` from `marrow/tests/`. Absolute `from marrow.x import` fails with `unable to locate module 'marrow'` when compiled as part of the package.
- **Import explicitly, never `from .x import *`.**
- **No `def main()` in any file under `marrow/`** — `mojo precompile marrow` rejects it. Test files are plain importable modules; the harness generates the driver.
- **Test case names must be unique across the entire suite**, not just per file.
- **`mojo precompile marrow` must stay at 0 errors and 0 warnings.** Verify with `pixi run -e dev precompile` (~18 s).
- **Never run `mojo test`, `mojo run`, or a hand-written driver.** Always go through `pytest`.
- **Binary size is reported as the `__text` section**, via `size -m <binary>` → `Section __text`. Never file size — it is quantized to 16,384 B on Apple Silicon.
- **Invariant 1 (small-binary DCE):** preserve closed erasure. Gate on `pixi run binary_size`.
- **Invariant 2 (one engine, two drivers):** no feature may exist in only one lane. Enforced by `marrow/exprold/tests/test_parity.mojo`.
- **Prefer typed aliases** (`Int32Array` over `PrimitiveArray[Int32Type]`) and typed shorthands (`.as_int32()`). Never `PrimitiveArray[bool_]` — booleans are bit-packed, use `BoolArray`.
- **`unsafe_ptr()` is restricted** to `buffers.mojo`, `views.mojo`, `c_data.mojo`, `utils/byteorder.mojo` and the Parquet codec layer. Nothing in this plan may call it.
- **Conventional commits** (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`), optional scope: `feat(expr): add param nodes`.
- **Add a `CHANGELOG.md` entry** under `## [Unreleased]` for every meaningful change.
- **Compilation is slow.** One pytest selection is one compilation unit; a directory of expr tests took 4 min 43 s. Select narrowly and expect minutes, not seconds.

---

## File Structure

**Created:**
- `marrow/exprold/params.mojo` — `ParamCell`, `ParamDecl`, the registry, and `PathSpec`. Pure data and lookup; imports only `dtypes` and `scalars`. No node types, no relational types, so it sits at the bottom of the `expr` import order and creates no cycle.
- `marrow/exprold/tests/test_params.mojo` — cells, registry drain, argv parsing, binding.
- `benchmarks/binary_size/query_param.mojo` — the invariant-1 gate.
- `python/marrow/compile.py` — the CLI.
- `python/marrow/tests/test_compile.py` — CLI unit tests (path resolution, arg construction) that do **not** invoke the compiler.

**Modified:**
- `marrow/exprold/values.mojo` — add `NumericParam[T]`, `StringParam[T]`, `TemporalParam[T]` beside their `*Literal` counterparts (`NumericLiteral` at `:831`, `StringLiteral` at `:1793`).
- `marrow/exprold/dynamic.mojo` — add `DynValue.param()` and its `_param` eval function beside `literal` at `:909`.
- `marrow/exprold/builders.mojo` — add the `param()` overload set. **It must live here**: this module's docstring records that an overload set cannot span modules, and splitting `col`/`lit` was reverted for exactly that reason (backlog L2).
- `marrow/exprold/relations.mojo` — `ParquetScan.path: String` → `PathSpec` (`:1033`), and `execute_cli()` on `DynRelation` beside `execute()` (`:390`).
- `marrow/exprold/tests/test_parity.mojo` — the invariant-2 case.
- `python/pyproject.toml` — console script, `compile` extra, wheel payload.
- `python/build.py` — force-include `marrow/_mojo/`.
- `CHANGELOG.md`, `docs/backlog.md`.

---

### Task 1: Parameter cells and the registry

**Files:**
- Create: `marrow/exprold/params.mojo`
- Create: `marrow/exprold/tests/test_params.mojo`

**Interfaces:**
- Consumes: `DynType` and `DynScalar` from `marrow.dtypes` / `marrow.scalars`.
- Produces:
  - `struct ParamCell(Copyable, Movable)` with
    `def __init__(out self, var name: String = String())`,
    `def get(self) raises -> DynScalar` (raises when unbound),
    `def set(mut self, var v: DynScalar)`, `def is_bound(self) -> Bool`,
    and `def name_hint(self) -> String` (used by `NumericParam.render`).
    **The name argument must default**, so both `ParamCell()` and
    `ParamCell(name.copy())` compile — the tests below use each.
  - `struct ParamDecl(Copyable, Movable)` with fields `name: String`,
    `dtype: DynType`, `help: String`, `default: Optional[DynScalar]`,
    `cell: ArcPointer[ParamCell]`, and `def is_required(self) -> Bool`.
    Its constructor is
    `def __init__(out self, *, var name: String, dtype: DynType, var help: String = String(), var default: Optional[DynScalar] = None, cell: Optional[ArcPointer[ParamCell]] = None)`;
    when `cell` is omitted it allocates `ArcPointer(ParamCell(name.copy()))`.
    **Every optional argument here is load-bearing** — Tasks 1, 4 and 6
    construct a `ParamDecl` from name and dtype alone.
  - `def register_param(var decl: ParamDecl)` — appends to the module registry.
  - `def drain_params() -> List[ParamDecl]` — returns the registry contents and empties it.
  - `struct PathSpec(Copyable, Movable)` with `@implicit def __init__(out self, var path: String)`, `def __init__(out self, cell: ArcPointer[ParamCell])`, and `def resolve(self) raises -> String`.

**Why a registry and not a `parameters()` trait method:** the spec's §3 records the measurement. A `parameters()` sibling to `referenced_columns()` would need **40 implementations** and a second recursive traversal in every node, against a size gate where one shared adapter already cost +662,740 bytes. Do not "improve" this into a trait method.

- [ ] **Step 1: Write the failing test**

Create `marrow/exprold/tests/test_params.mojo`:

```mojo
from std.testing import assert_true, assert_false, assert_raises
from std.memory import ArcPointer
from ...dtypes import DynType, int64, string
from ...scalars import Int64Scalar, StringScalar
from ..params import (
    ParamCell,
    ParamDecl,
    PathSpec,
    drain_params,
    register_param,
)


def test_param_cell_unbound_raises() raises:
    var cell = ParamCell()
    assert_false(cell.is_bound())
    with assert_raises():
        _ = cell.get()


def test_param_cell_binds_and_reads() raises:
    var cell = ParamCell()
    cell.set(Int64Scalar(7).to_dyn())
    assert_true(cell.is_bound())
    assert_true(cell.get().as_int64().value() == 7)


def test_param_registry_drains_empty() raises:
    _ = drain_params()
    assert_true(len(drain_params()) == 0)


def test_param_registry_drains_once() raises:
    _ = drain_params()
    register_param(
        ParamDecl(name="src", dtype=DynType(string), help=String("in"))
    )
    var first = drain_params()
    assert_true(len(first) == 1)
    assert_true(first[0].name == "src")
    assert_true(len(drain_params()) == 0)


def test_path_spec_literal_resolves() raises:
    var spec = PathSpec(String("a.parquet"))
    assert_true(spec.resolve() == "a.parquet")


def test_path_spec_cell_resolves() raises:
    var cell = ArcPointer(ParamCell())
    var spec = PathSpec(cell)
    cell[].set(StringScalar(String("b.parquet")).to_dyn())
    assert_true(spec.resolve() == "b.parquet")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pixi run -e dev pytest marrow/exprold/tests/test_params.mojo -v`
Expected: FAIL — `unable to locate module` for `..params`. This takes minutes; that is normal.

- [ ] **Step 3: Implement `marrow/exprold/params.mojo`**

Write the module against the interfaces above. Requirements the tests encode:
- `ParamCell` holds `Optional[DynScalar]`; `get()` raises a message naming the parameter is *not* possible here (the cell does not know its name) — raise `"parameter is not bound"` and let `ParamDecl` produce the named error.
- The registry is a module-level `List[ParamDecl]`. `drain_params()` moves it out and leaves an empty list.
- `PathSpec` holds a `Variant[String, ArcPointer[ParamCell]]`; `resolve()` returns the literal or reads the cell's `as_string().value()`.
- `ParamDecl.is_required()` is `not self.default`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `pixi run -e dev pytest marrow/exprold/tests/test_params.mojo -v`
Expected: 6 PASS.

- [ ] **Step 5: Verify the tree is warning-clean**

Run: `pixi run -e dev precompile`
Expected: 0 errors, 0 warnings. A new warning here is a task failure, not a follow-up.

- [ ] **Step 6: Commit**

```bash
git add marrow/exprold/params.mojo marrow/exprold/tests/test_params.mojo
git commit -m "feat(expr): add parameter cells, declarations and the registry"
```

---

### Task 2: `NumericParam[T]` and the numeric `param()` overload

**Files:**
- Modify: `marrow/exprold/values.mojo` (add beside `NumericLiteral` at `:831`)
- Modify: `marrow/exprold/builders.mojo` (add beside `lit` at `:63`)
- Modify: `marrow/exprold/tests/test_params.mojo`

**Interfaces:**
- Consumes: `ParamCell`, `ParamDecl`, `register_param` from Task 1.
- Produces:
  - `struct NumericParam[T: NumericType](NumericValue)` with `var _cell: ArcPointer[ParamCell]`.
  - `def param[T: NumericType](var name: String, dtype: T, default: Optional[Int] = None, var help: String = String()) -> NumericParam[T]`.

**The exemplar is `NumericLiteral[T]` at `values.mojo:831`.** Copy its member set exactly — `OutType`, `OutShape`, `State`, `render`, `prune`, `referenced_columns`, `state`, `lane` — and change only where noted. Everything else (`validity`, `bound_column`, `name`) is inherited from the `Value` defaults at `:398`, `:420`, `:444`.

- [ ] **Step 1: Write the failing test**

Append to `marrow/exprold/tests/test_params.mojo`:

```mojo
def test_numeric_param_binds_into_a_fused_predicate() raises:
    _ = drain_params()
    var a = array([1, 5, 3, 8, 2], int64)
    var batch = record_batch([a.copy()], names=["a"])

    var min_a = param("min-a", int64)
    var pred = col("a", int64) > min_a

    var decls = drain_params()
    assert_true(len(decls) == 1)
    decls[0].cell[].set(Int64Scalar(3).to_dyn())

    var out = BoxedValue(pred).execute(batch)
    assert_true(out.as_bool() == array([False, True, False, True, False]))


def test_numeric_param_unbound_raises_at_execute() raises:
    _ = drain_params()
    var a = array([1, 2], int64)
    var batch = record_batch([a.copy()], names=["a"])
    var pred = col("a", int64) > param("min-a", int64)
    with assert_raises():
        _ = BoxedValue(pred).execute(batch)
```

Add the imports these need: `array` from `...builders`, `record_batch` from `...tabular`, `col`/`param` from `..builders`, `BoxedValue` from `..values`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `pixi run -e dev pytest marrow/exprold/tests/test_params.mojo -v`
Expected: FAIL — `param` is undefined.

- [ ] **Step 3: Add `NumericParam[T]` to `values.mojo`**

```mojo
struct NumericParam[T: NumericType](NumericValue):
    """A numeric value supplied at run time, broadcast into every lane.

    Structurally `NumericLiteral[T]` with the scalar behind a cell. Resolving
    the cell in `state()` rather than `lane()` is what makes this free: the lane
    splats a plain `Scalar`, byte-identical to a literal's, so a parameter costs
    nothing per row."""

    comptime OutType = Self.T
    comptime OutShape = 0
    comptime State = Scalar[Self.OutType.native]

    var _cell: ArcPointer[ParamCell]

    def render(self) -> String:
        return String("param(", self._cell[].name_hint(), ")")

    def referenced_columns(self) -> List[String]:
        return List[String]()

    def prune(self, stats: PruneStats) raises -> Interval:
        var v = self._cell[].get()
        return Interval.bounds(Optional(v.copy()), Optional(v^))

    def state(self, batch: RecordBatch) raises -> Self.State:
        return self._cell[].get().as_primitive[Self.T]().value()

    @always_inline
    def lane[
        W: Int
    ](self, state: Self.State, idx: Int) -> SIMD[Self.OutType.native, W]:
        return SIMD[Self.OutType.native, W](state)
```

`prune()` reading the cell is deliberate — a bound parameter must prune row groups exactly as a literal does. If the cell is unbound, `get()` raises, which is correct: pruning cannot run before binding.

Add `name_hint() -> String` to `ParamCell` in Task 1's module (store the name on the cell) so `render()` can print something useful.

- [ ] **Step 4: Add the `param()` overload to `builders.mojo`**

```mojo
def param[
    T: NumericType
](
    var name: String,
    dtype: T,
    default: Optional[Int] = None,
    var help: String = String(),
) -> NumericParam[T]:
    """A numeric value supplied at run time — `param("min-a", int64)`."""
    var cell = ArcPointer(ParamCell(name.copy()))
    var dflt = Optional[DynScalar](None)
    if default:
        dflt = Optional(PrimitiveScalar[T](default.value()).to_dyn())
    register_param(
        ParamDecl(
            name=name.copy(),
            dtype=DynType(dtype),
            help=help^,
            default=dflt^,
            cell=cell,
        )
    )
    return NumericParam[T](cell)
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `pixi run -e dev pytest marrow/exprold/tests/test_params.mojo -v`
Expected: 8 PASS.

- [ ] **Step 6: Verify warning-clean and commit**

```bash
pixi run -e dev precompile   # 0 errors, 0 warnings
git add marrow/exprold/values.mojo marrow/exprold/builders.mojo marrow/exprold/params.mojo marrow/exprold/tests/test_params.mojo
git commit -m "feat(expr): add NumericParam and the numeric param() builder"
```

---

### Task 3: `StringParam[T]` and `TemporalParam[T]`

**Files:**
- Modify: `marrow/exprold/values.mojo` (beside `StringLiteral` at `:1793`, and beside the temporal nodes near `:2452`)
- Modify: `marrow/exprold/builders.mojo`
- Modify: `marrow/exprold/tests/test_params.mojo`

**Interfaces:**
- Produces: `StringParam[T: StringLikeType](StringValue)`, `TemporalParam[T: TemporalType](TemporalValue)`, and the matching `param()` overloads returning them.

**Note the family differences.** `StringValue.lane` has **no `[W]` parameter** — variable-width UTF-8 has no SIMD lane, so it is `def lane(self, state: Self.State, idx: Int) -> String` (see the trait at `values.mojo:1633`). `StringParam.State` is `String`; `TemporalParam` mirrors `NumericParam` with `State = Scalar[Self.OutType.native]`.

`param("src", string)` is what the scan path uses, so this task unblocks Task 5.

- [ ] **Step 1: Write the failing tests**

```mojo
def test_string_param_binds_into_a_fused_predicate() raises:
    _ = drain_params()
    var s = array(["p", "q", "p"])
    var batch = record_batch([s.copy()], names=["s"])
    var want = param("want", string)
    var pred = col("s", string) == want
    var decls = drain_params()
    decls[0].cell[].set(StringScalar(String("p")).to_dyn())
    var out = BoxedValue(pred).execute(batch)
    assert_true(out.as_bool() == array([True, False, True]))


def test_string_param_default_is_used_when_unset() raises:
    _ = drain_params()
    _ = param("want", string, default=String("q"))
    var decls = drain_params()
    assert_true(decls[0].default.value().as_string().value() == "q")
    assert_false(decls[0].is_required())
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run -e dev pytest marrow/exprold/tests/test_params.mojo -v`
Expected: FAIL — no `param` overload for `StringLikeType`.

- [ ] **Step 3: Implement both nodes and both overloads**

Mirror Task 2 exactly, changing the family trait, the `State` type, the `lane` signature (no `[W]` for strings), and the default's scalar type (`StringScalar` / `PrimitiveScalar[T]`).

- [ ] **Step 4: Run to verify pass**

Run: `pixi run -e dev pytest marrow/exprold/tests/test_params.mojo -v`
Expected: 10 PASS.

- [ ] **Step 5: Verify warning-clean and commit**

```bash
pixi run -e dev precompile
git add marrow/exprold/values.mojo marrow/exprold/builders.mojo marrow/exprold/tests/test_params.mojo
git commit -m "feat(expr): add StringParam and TemporalParam"
```

---

### Task 4: The runtime-lane parameter — invariant 2

**Files:**
- Modify: `marrow/exprold/dynamic.mojo` (beside `literal` at `:909` and `_literal` at `:338`)
- Modify: `marrow/exprold/builders.mojo`
- Modify: `marrow/exprold/tests/test_parity.mojo`

**Interfaces:**
- Produces: `DynValue.param(var name: String) -> Self` (a `@staticmethod`, matching `column` at `:905` and `literal` at `:909`), and a `param(var name: String, dtype: DynType) -> DynValue` overload in `builders.mojo`.

**Do not widen `DynPayload`.** It is `Variant[NoneType, String, DynType, DynArray, DynScalar]` at `dynamic.mojo:223` and is inline, so adding a member grows every `DynValue`. Use the **existing `String` arm** to carry the parameter name and have `_param` look the value up in the registry by name — the same shape `column` already uses. The lookup runs once per batch.

**Invariant 2 is not optional.** A feature may not exist in only one lane; `test_parity.mojo` enforces it across four axes.

- [ ] **Step 1: Write the failing parity test**

Add to `marrow/exprold/tests/test_parity.mojo`, following the file's existing case style:

```mojo
def test_parity_param_both_lanes_agree() raises:
    var a = array([1, 5, 3, 8, 2], int64)
    var batch = record_batch([a.copy()], names=["a"])

    _ = drain_params()
    var fused = col("a", int64) > param("min-a", int64)
    var fused_decls = drain_params()
    fused_decls[0].cell[].set(Int64Scalar(3).to_dyn())
    var fused_out = BoxedValue(fused).execute(batch)

    _ = drain_params()
    var dyn = col("a") > param("min-a", DynType(int64))
    var dyn_decls = drain_params()
    dyn_decls[0].cell[].set(Int64Scalar(3).to_dyn())
    var dyn_out = dyn.execute(batch)

    assert_true(fused_out.as_bool() == dyn_out.as_bool())
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run -e dev pytest marrow/exprold/tests/test_parity.mojo -v`
Expected: FAIL — no `param` overload taking a `DynType`.

- [ ] **Step 3: Implement `_param`, `DynValue.param`, and the builder overload**

```mojo
    @staticmethod
    def _param(
        args: List[DynArray], payload: DynPayload, batch: RecordBatch
    ) raises -> DynArray:
        return lookup_param(payload[String]).get().repeat(batch.num_rows())

    @staticmethod
    def param(var name: String) -> Self:
        return Self("param", Self._param, DynPayload(name^))
```

Add `def lookup_param(name: String) raises -> ParamCell` to `marrow/exprold/params.mojo`, searching a second module-level list that `register_param` also appends to and that `drain_params` does **not** clear — the runtime lane resolves by name at execute time, after the declarations have been drained. Document that asymmetry in the module docstring.

- [ ] **Step 4: Run to verify pass**

Run: `pixi run -e dev pytest marrow/exprold/tests/test_parity.mojo marrow/exprold/tests/test_params.mojo -v`
Expected: all PASS.

- [ ] **Step 5: Verify warning-clean and commit**

```bash
pixi run -e dev precompile
git add marrow/exprold/dynamic.mojo marrow/exprold/builders.mojo marrow/exprold/params.mojo marrow/exprold/tests/test_parity.mojo
git commit -m "feat(expr): add the runtime-lane param leaf, at parity with the fused one"
```

---

### Task 5: `PathSpec` on `ParquetScan`

**Files:**
- Modify: `marrow/exprold/relations.mojo:1033-1049` (the field and `__init__`), `:1060` and `:1085` (the two rebuild sites in `with_predicate` / `with_projection`)
- Modify: `marrow/exprold/tests/test_params.mojo`

**Interfaces:**
- Consumes: `PathSpec` from Task 1.
- Produces: `ParquetScan.path: PathSpec`, still constructible from a bare `String` via the `@implicit` conversion.

**Every existing call site must keep compiling untouched.** `PathSpec.__init__(out self, var path: String)` is `@implicit`, so `ParquetScan(path=String("orders.parquet"), ...)` continues to work. Verify that claim rather than assuming it — the gates in `benchmarks/binary_size/` and the Parquet tests are the proof.

**The layout constraint does not block this.** `CLAUDE.md`'s "do not change layout" names **array, scalar and builder** layout; `ParquetScan` is a relation node. The backlog explicitly records that this constraint has been read too broadly before (the B13 case).

- [ ] **Step 1: Write the failing test**

```mojo
def test_parquet_scan_accepts_a_param_path() raises:
    _ = drain_params()
    var src = param("src", string)
    var scan = ParquetScan[LeafSet.all()](
        path=src, schema=schema([field("a", int64)])
    )
    var decls = drain_params()
    decls[0].cell[].set(StringScalar(String("x.parquet")).to_dyn())
    assert_true(scan.path.resolve() == "x.parquet")
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run -e dev pytest marrow/exprold/tests/test_params.mojo -v`
Expected: FAIL — `path` does not accept a `StringParam`.

- [ ] **Step 3: Change the field to `PathSpec`**

Add a `PathSpec.__init__` overload taking a `StringParam[T]` (it carries the cell), change the field type, and update the two rebuild sites to pass `self.path.copy()`. Replace every internal read of `self.path` with `self.path.resolve()` — the reader opens the file at processor-build time, which is after binding.

- [ ] **Step 4: Run the test and the Parquet + size-gate call sites**

Run: `pixi run -e dev pytest marrow/exprold/tests/test_params.mojo marrow/parquet/tests -v`
Expected: all PASS, with no change required to any existing `ParquetScan(path=String(...))` call.

- [ ] **Step 5: Verify warning-clean and commit**

```bash
pixi run -e dev precompile
git add marrow/exprold/relations.mojo marrow/exprold/params.mojo marrow/exprold/tests/test_params.mojo
git commit -m "feat(expr): let ParquetScan take a late-bound path"
```

---

### Task 6: `execute_cli()`

**Files:**
- Modify: `marrow/exprold/relations.mojo` (add beside `execute` at `:390`)
- Modify: `marrow/exprold/params.mojo` (the argv parser)
- Modify: `marrow/exprold/tests/test_params.mojo`

**Interfaces:**
- Produces:
  - `def parse_params(args: List[String], decls: List[ParamDecl]) raises` in `params.mojo` — binds each declaration's cell from `--name value` pairs, applies defaults, raises a named error for a missing required parameter or an unknown flag.
  - `def render_usage(decls: List[ParamDecl]) -> String` and `def render_describe(decls: List[ParamDecl]) -> String` (JSON).
  - `def execute_cli(self, ctx: ExecContext = ExecContext.auto()) raises` on `DynRelation`.

**Order of operations in `execute_cli()`:** drain the registry → if `--help` or `--describe` is present, print and **return without executing** → `parse_params` → `execute(ctx)` → write output. Short-circuiting before execution is what removes any need for dummy values; the plan is built with unbound cells and simply never runs.

**Output contract:** no `-o` pretty-prints to stdout; `-o r.parquet` writes Parquet; `-o r.arrow` writes Arrow IPC; `--format parquet|ipc|table` overrides the extension.

**Measure the writers before making them unconditional.** Spec open question 2: if linking the Parquet and IPC writers is expensive on the size gate, make each opt-in. Task 7 produces the number; if it is large, come back and gate them.

- [ ] **Step 1: Write the failing tests for the parser**

```mojo
def test_parse_params_binds_by_name() raises:
    var decls = List[ParamDecl]()
    decls.append(ParamDecl(name="min-a", dtype=DynType(int64)))
    parse_params(["--min-a", "5"], decls)
    assert_true(decls[0].cell[].get().as_int64().value() == 5)


def test_parse_params_applies_defaults() raises:
    var decls = List[ParamDecl]()
    decls.append(
        ParamDecl(
            name="min-a",
            dtype=DynType(int64),
            default=Optional(Int64Scalar(9).to_dyn()),
        )
    )
    parse_params(List[String](), decls)
    assert_true(decls[0].cell[].get().as_int64().value() == 9)


def test_parse_params_missing_required_raises() raises:
    var decls = List[ParamDecl]()
    decls.append(ParamDecl(name="src", dtype=DynType(string)))
    with assert_raises():
        parse_params(List[String](), decls)


def test_parse_params_unknown_flag_raises() raises:
    var decls = List[ParamDecl]()
    with assert_raises():
        parse_params(["--nope", "1"], decls)


def test_render_usage_names_every_param() raises:
    var decls = List[ParamDecl]()
    decls.append(
        ParamDecl(name="src", dtype=DynType(string), help=String("input"))
    )
    var usage = render_usage(decls)
    assert_true("--src" in usage)
    assert_true("input" in usage)
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run -e dev pytest marrow/exprold/tests/test_params.mojo -v`
Expected: FAIL — `parse_params` undefined.

- [ ] **Step 3: Implement the parser, the renderers, and `execute_cli`**

`parse_params` takes an explicit `List[String]` rather than reading `argv` so it is testable without a process; `execute_cli` supplies the real `argv`.

- [ ] **Step 4: Run to verify pass**

Run: `pixi run -e dev pytest marrow/exprold/tests/test_params.mojo -v`
Expected: all PASS.

- [ ] **Step 5: Verify warning-clean and commit**

```bash
pixi run -e dev precompile
git add marrow/exprold/params.mojo marrow/exprold/relations.mojo marrow/exprold/tests/test_params.mojo
git commit -m "feat(expr): add execute_cli with argv binding, --help and --describe"
```

---

### Task 7: The binary-size gate — invariant 1

**Files:**
- Create: `benchmarks/binary_size/query_param.mojo`
- Modify: `benchmarks/binary_size/compare.py` (add `"query_param"` to `NAMES`)
- Modify: `benchmarks/binary_size/README.md`

**Interfaces:**
- Consumes: everything from Tasks 2-6.

**Model it on `query_scan_typed.mojo`** — identical query and plan, with the literal replaced by a `param()` and the path late-bound. The `__text` delta against `query_scan_typed` is then exactly what parameters cost. **Expect near-zero**: a parameter is a literal plus a pointer hop resolved once per batch.

**Read the number correctly.** `size -m <binary>` → `Section __text`. File size is quantized to 16,384 B on Apple Silicon and will lie to you: a real +1,728-byte change once showed as +16,504 with one *fewer* symbol.

- [ ] **Step 1: Write `query_param.mojo`**

Copy `benchmarks/binary_size/query_scan_typed.mojo` verbatim, then replace the scan path with `param("src", string)` and the comparison's right operand with `param("min-a", int64)`, and end with `plan.execute_cli()` instead of `print(...execute())`.

- [ ] **Step 2: Build and measure both gates**

```bash
pixi run binary_size query_param query_scan_typed
```

Expected: a table including both. Record the `__text` delta.

- [ ] **Step 3: Judge the result**

If `query_param` − `query_scan_typed` is more than ~20 KB, stop and investigate before continuing: the likely cause is the Parquet/IPC writers linked by `execute_cli`'s output path (spec open question 2), in which case make each writer opt-in behind a comptime flag and re-measure.

- [ ] **Step 4: Record the baseline and commit**

Add the row to the README table with the measured number and today's date, then:

```bash
git add benchmarks/binary_size/query_param.mojo benchmarks/binary_size/compare.py benchmarks/binary_size/README.md
git commit -m "test(binary_size): gate what a late-bound parameter costs"
```

---

### Task 8: The `marrow compile` CLI

**Files:**
- Create: `python/marrow/compile.py`
- Create: `python/marrow/tests/test_compile.py`
- Modify: `python/pyproject.toml`

**Interfaces:**
- Produces:
  - `def resolve_marrow_path(explicit: str | None = None) -> Path` — resolution order `--marrow-path` → `$MARROW_MOJO_PATH` → the bundled `marrow/_mojo/` → repo-root autodetect. Raises `FileNotFoundError` with all four locations listed when none resolve.
  - `def build_command(src: Path, out: Path, marrow_path: Path, opt: str = "-O3") -> list[str]`.
  - `def check_mojo_version() -> str` — raises `RuntimeError` naming the required range when `mojo` is absent or out of range.
  - `def main(argv: list[str] | None = None) -> int` — the console-script entry.

**The build recipe already exists.** `benchmarks/binary_size/compare.py:build_and_strip` is the working version: `mojo build -O3 -g0 -I . <src> -o <out>`, then `strip`. Reuse its shape.

**Unit-test the pure functions only.** Invoking the compiler in a unit test costs 1-2 minutes per case; the tests here must not do it.

- [ ] **Step 1: Write the failing tests**

```python
import pytest
from pathlib import Path
from marrow.compile import build_command, resolve_marrow_path


def test_build_command_uses_o3_and_include_path(tmp_path):
    cmd = build_command(tmp_path / "q.mojo", tmp_path / "q", tmp_path / "src")
    assert cmd[:2] == ["mojo", "build"]
    assert "-O3" in cmd and "-g0" in cmd
    assert cmd[cmd.index("-I") + 1] == str(tmp_path / "src")
    assert cmd[cmd.index("-o") + 1] == str(tmp_path / "q")


def test_resolve_marrow_path_prefers_explicit(tmp_path):
    (tmp_path / "marrow").mkdir()
    assert resolve_marrow_path(str(tmp_path)) == tmp_path


def test_resolve_marrow_path_reads_env(tmp_path, monkeypatch):
    (tmp_path / "marrow").mkdir()
    monkeypatch.setenv("MARROW_MOJO_PATH", str(tmp_path))
    assert resolve_marrow_path() == tmp_path


def test_resolve_marrow_path_reports_every_location(tmp_path, monkeypatch):
    monkeypatch.delenv("MARROW_MOJO_PATH", raising=False)
    with pytest.raises(FileNotFoundError) as exc:
        resolve_marrow_path(str(tmp_path / "nope"))
    assert "MARROW_MOJO_PATH" in str(exc.value)
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run -e dev pytest python/marrow/tests/test_compile.py -v`
Expected: FAIL — `No module named marrow.compile`.

- [ ] **Step 3: Implement `python/marrow/compile.py`**

`argparse`-based, standard library only — the repo's guidance is to avoid unnecessary dependencies. Flags: `<file>`, `[out]`, `--marrow-path`, `--fast` (`-O1`), `--no-strip`, `--bundle DIR` (Task 9), `-v`.

- [ ] **Step 4: Run to verify pass**

Run: `pixi run -e dev pytest python/marrow/tests/test_compile.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Register the console script**

In `python/pyproject.toml`:

```toml
[project.scripts]
marrow = "marrow.compile:main"

[project.optional-dependencies]
compile = ["mojo>=1.1,<2"]
```

**Record the known limitation in the CLI's error message**: marrow pins a Mojo *nightly* (1.1.0.dev2026081705) while PyPI stable is 1.0.0, and a wheel cannot force an `--extra-index-url`. `check_mojo_version` must say so plainly rather than letting an opaque compiler error surface.

- [ ] **Step 6: End-to-end smoke test, run once by hand**

```bash
pixi run python -m marrow.compile benchmarks/binary_size/query_param.mojo -o /tmp/qp
/tmp/qp --help
```

Expected: usage listing `--src` and `--min-a`. Expect the build to take 1-2 minutes.

- [ ] **Step 7: Commit**

```bash
git add python/marrow/compile.py python/marrow/tests/test_compile.py python/pyproject.toml
git commit -m "feat(cli): add marrow compile"
```

---

### Task 9: `--bundle`

**Files:**
- Modify: `python/marrow/compile.py`
- Modify: `python/marrow/tests/test_compile.py`

**Interfaces:**
- Produces:
  - `def dylib_closure(binary: Path) -> list[Path]` — the **transitive** closure via repeated `otool -L` (macOS) or `ldd` (Linux), excluding `/usr/lib` and `/System`.
  - `def bundle(binary: Path, dest: Path) -> Path` — copies the binary and its closure into `dest` and rewrites the rpath to `@loader_path` (macOS, `install_name_tool`) or `$ORIGIN` (Linux, `patchelf`).

**Static linking is impossible — do not retry it.** The spec's §7 records four probes: `-Xlinker -static` fails with `ld: library 'System' not found` (macOS ships no `libSystem.a`), `-Xlinker -Bstatic` is not an Apple flag, `-dead_strip_dylibs` works but strips nothing from a real query (47 undefined symbols: 22 libc, 15 KGEN, 10 AsyncRT), and the shipped `.dylib`s cannot become `.a` files — they are fully linked Mach-O images with no embedded bitcode.

**Walk the closure, never hardcode it.** The binary links **2** dylibs directly but depends on **4** transitively. Modular's own list also includes `MGPRT` (the GPU runtime), which a `-D MARROW_GPU=true` build would likely add as a 5th.

- [ ] **Step 1: Write the failing test**

```python
def test_dylib_closure_is_transitive_and_excludes_system():
    from marrow.compile import dylib_closure
    binary = Path("benchmarks/binary_size/query_scan_typed")
    if not binary.exists():
        pytest.skip("gate binary not built")
    names = {p.name for p in dylib_closure(binary)}
    assert "libAsyncRTMojoBindings.dylib" in names
    assert "libAsyncRTRuntimeGlobals.dylib" in names  # transitive, not direct
    assert not any(n.startswith("libSystem") for n in names)
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run -e dev pytest python/marrow/tests/test_compile.py -v`
Expected: FAIL — `dylib_closure` undefined.

- [ ] **Step 3: Implement `dylib_closure` and `bundle`**

- [ ] **Step 4: Run to verify pass, then verify the bundle actually runs**

```bash
pixi run python -m marrow.compile benchmarks/binary_size/query_param.mojo -o /tmp/qp --bundle /tmp/qpdir
cd /tmp/qpdir && ./query_param --help
otool -l ./query_param | grep -A2 LC_RPATH   # must be @loader_path, not an absolute pixi path
```

Expected: usage prints, and the rpath is `@loader_path`. **This is the whole point of the task** — an absolute rpath means the artifact does not run off the build machine.

- [ ] **Step 5: Commit**

```bash
git add python/marrow/compile.py python/marrow/tests/test_compile.py
git commit -m "feat(cli): bundle the transitive dylib closure with a relocatable rpath"
```

---

### Task 10: Wheel payload

**Files:**
- Modify: `python/build.py`
- Modify: `python/pyproject.toml`

**Interfaces:**
- Produces: `marrow/_mojo/marrow/**/*.mojo` inside the wheel, so `resolve_marrow_path()` finds `<site-packages>/marrow/_mojo`.

**Ship source, not `.mojoc`.** Measured: source without tests is **1,678,678 B** against the `.mojoc`'s **5,625,382 B** — 3.4x smaller — and strictly more tolerant, since a `.mojoc` is pegged to the exact compiler build and hard-errors on any skew. Both are architecture-independent.

**One wheel, not two.** Extras add dependencies, not files. 1.68 MB on an already-21 MB wheel does not justify a second `marrow-mojo` distribution and its lockstep-release skew risk.

- [ ] **Step 1: Exclude tests and benches from the payload**

In `python/build.py`, extend the hook to walk `marrow/**/*.mojo`, skipping any path containing `/tests/` or a `bench_`/`profile_` filename, and add each to `build_data["force_include"]` under `marrow/_mojo/marrow/...`.

- [ ] **Step 2: Verify the payload size**

```bash
find marrow -name '*.mojo' -not -path '*/tests/*' -exec cat {} + | wc -c
```

Expected: about 1,678,678 — if it is near 2,832,303 the test exclusion is not working.

- [ ] **Step 3: Build the wheel and inspect it**

```bash
pixi run -e wheel wheel
python -c "import zipfile,glob; z=zipfile.ZipFile(glob.glob('dist/*.whl')[0]); print(sum(1 for n in z.namelist() if n.startswith('marrow/_mojo/')), 'mojo files')"
```

Expected: a nonzero count, and no `tests/` entries.

- [ ] **Step 4: Commit**

```bash
git add python/build.py python/pyproject.toml
git commit -m "build: ship marrow's Mojo source in the wheel for marrow compile"
```

---

### Task 11: Documentation and backlog

**Files:**
- Create: `docs/guide/compile.qmd`
- Modify: `docs/guide/expressions.qmd`, `CHANGELOG.md`, `docs/backlog.md`

**This closes M1.6.** The backlog records that `docs/guide/expressions.qmd`'s AOT blocks are illustrative (plain ` ```python `, never executed), "which is why they could name types that did not exist and the docs build stayed green — an executed example is the only thing that keeps this page honest." The example added here must be **runnable**.

- [ ] **Step 1: Write `docs/guide/compile.qmd`**

Cover: the `param()` surface, `marrow compile`, `--help` / `--describe`, `-o`, `--bundle`. **State the compile-time cost honestly** — roughly 1-2 minutes per invocation, since every build elaborates all of marrow at `-O3`. Users must not discover that themselves.

- [ ] **Step 2: Build the docs**

Run: `pixi run -e docs docs`
Expected: green, with the new page rendered.

- [ ] **Step 3: Add the CHANGELOG entry**

Under `## [Unreleased]` → `### Features`: the `param()` nodes in both lanes, `execute_cli()`, and the `marrow compile` CLI with `--bundle`.

- [ ] **Step 4: File the follow-up backlog card**

Add a card recording the spec's §7 finding: all 10 `_AsyncRT_*` symbols in a **GPU-off** binary are `DeviceBuffer`/`DeviceContext` device calls. Eliminating them would let `-Xlinker -dead_strip_dylibs` drop `libAsyncRTMojoBindings.dylib` entirely — **1,156,592 B, 42% of the runtime closure**. Note that CLAUDE.md already records the symptom ("a GPU-off binary still links it"); this is the cause.

- [ ] **Step 5: Mark M1.6 done and commit**

```bash
git add docs/ CHANGELOG.md
git commit -m "docs: add the marrow compile guide and a runnable AOT example"
```

---

## Optional probes (spec open questions 1 and 3)

Neither blocks delivery; both are cheap and answer a question the spec
left open. Run them if time allows and record the numbers in the spec.

- **Does `.mojoc` beat source for `marrow compile` wall-clock?** Build the
  same query twice — once with `-I <source tree>`, once against
  `.precompile/marrow.mojoc` — and compare cold wall-clock. Expect no
  meaningful difference: `.mojoc` skips parsing, not elaboration, and
  elaboration is the cost. A null result is a real result; write it down.
- **Does `-Xlinker -static` work on Linux?** It cannot be tested on the
  darwin development box, and Linux is the only platform where static
  linking could ever work. If it does, `--static` becomes a flag over the
  same code path as `--bundle`.

## Verification before calling this done

Run each and confirm the output, per `superpowers:verification-before-completion`:

```bash
pixi run -e dev precompile                                   # 0 errors, 0 warnings
pixi run -e dev pytest marrow/exprold/tests -v                  # includes params + parity
pixi run -e dev pytest marrow/parquet/tests -v               # PathSpec did not break the reader
pixi run -e dev pytest python/marrow/tests/test_compile.py -v
pixi run binary_size query_param query_scan_typed            # invariant 1
pixi run -e dev fmt                                          # mojo format + ruff format
```

Do not report completion on a partial run. If a step is skipped, say which and why.
