"""Viability skeleton for docs/lane-shape-window-design.md — the clean split, no
ExecCtx / cursor / per-node state. Compile & run: `mojo run lane-shape-window-skeleton.mojo`.

The design in one sentence: **fusable nodes implement `core` (and fuse); cross-row
nodes implement `execute` (and materialize).** A cross-row node computes and
*returns* a `ColumnarValue` — nothing is stored, so there is no `ExecCtx`, no
cursor, no slot, and no cross-batch staleness. Two traits capture it:

    trait Value:            execute(batch) -> ColumnarValue          # every node
    trait Fusable(Value):      core[W](batch, idx) -> SIMD              # + fuses

- `Fusable` nodes (columns, literals, arithmetic, and — via Angle 1 — len/parse/
  bool→num) inherit a fused `execute` default that runs one vectorized `core` loop.
- Cross-row nodes (`Sum`, `WindowFunction`) are `Value` only: their `execute`
  materializes and returns a `ColumnarValue`.
- Mixed arithmetic (a cross-row operand) is a `MatBinary` (Value): `execute`
  eagerly combines its children's `ColumnarValue`s with the kernel's `apply`.

The fuse/materialize boundary is the `Fusable`/`Value` trait line — resolved at
comptime by node type, not by a flag. `execute` returning a concrete
`ColumnarValue` (not `Self.ArrayType`) lets `Fusable` safely default the parent
trait's abstract `execute` with no associated-type recursion.

Covers the four load-bearing corners: ColumnarValue as the uniform result,
AnyValue erasure, a dynamic (interpreted) driver sharing the same kernels, and a
WindowFunction carrying a runtime spec with erased keys.
"""

from std.memory import ArcPointer

comptime i64 = DType.int64


# ===========================================================================
# ColumnarValue — Scalar | Array. `load[W]` hides splat-vs-load.
# ===========================================================================
@fieldwise_init
struct ColumnarValue(Copyable, Movable):
    var _is_scalar: Bool
    var _v: Int64
    var _a: List[Int64]

    @staticmethod
    def scalar(v: Int64) -> Self:
        return Self(True, v, List[Int64]())

    @staticmethod
    def columnar(var a: List[Int64]) -> Self:
        return Self(False, 0, a^)

    def is_scalar(self) -> Bool:
        return self._is_scalar

    def scalar_value(self) -> Int64:
        return self._v

    def num_rows(self) -> Int:
        return 1 if self._is_scalar else len(self._a)

    def into_array(self, n: Int) -> List[Int64]:  # lazy broadcast
        if self._is_scalar:
            var out = List[Int64](capacity=n)
            for _ in range(n):
                out.append(self._v)
            return out^
        return self._a.copy()


# ===========================================================================
# Kernels — one struct, two faces: core[W] (fused) + apply (eager)
# ===========================================================================
struct AddKernel:
    @always_inline
    @staticmethod
    def core[W: Int](a: SIMD[i64, W], b: SIMD[i64, W]) -> SIMD[i64, W]:
        return a + b

    @staticmethod
    def apply(a: ColumnarValue, b: ColumnarValue) raises -> ColumnarValue:
        if a.is_scalar() and b.is_scalar():
            return ColumnarValue.scalar(a.scalar_value() + b.scalar_value())
        var n = max(a.num_rows(), b.num_rows())
        var xa = a.into_array(n)
        var xb = b.into_array(n)
        var out = List[Int64](capacity=n)
        for i in range(n):
            out.append(xa[i] + xb[i])
        return ColumnarValue.columnar(out^)


trait WindowKernel:
    comptime frame_dependent: Bool

    @staticmethod
    def evaluate_all(values: List[Int64]) raises -> List[Int64]:
        ...


struct RowNumberKernel(WindowKernel):
    comptime frame_dependent = False

    @staticmethod
    def evaluate_all(values: List[Int64]) raises -> List[Int64]:
        var out = List[Int64](capacity=len(values))
        for i in range(len(values)):  # rank = 1 + count(v < self)
            var r = Int64(1)
            for j in range(len(values)):
                if values[j] < values[i]:
                    r += 1
            out.append(r)
        return out^


struct Batch(Copyable, Movable):
    var cols: List[List[Int64]]

    def __init__(out self, var cols: List[List[Int64]]):
        self.cols = cols^

    def num_rows(self) -> Int:
        return len(self.cols[0])


def load[W: Int](data: List[Int64], idx: Int) -> SIMD[i64, W]:
    var v = SIMD[i64, W](0)
    for k in range(W):
        v[k] = data[idx + k]
    return v


# ===========================================================================
# The two traits.
#   Value : every node — execute() -> ColumnarValue
#   Fusable  : + core[W]; inherits a fused execute default (uses core)
# ===========================================================================
trait Value(Copyable, ImplicitlyDeletable, Movable):
    comptime Shape: Int  # 0 scalar, 1 columnar

    def execute(self, batch: Batch) raises -> ColumnarValue:
        ...


trait Fusable(Value):
    def core[W: Int](self, batch: Batch, idx: Int) -> SIMD[i64, W]:
        ...

    # A subtrait may default the parent's abstract `execute` because it returns a
    # concrete ColumnarValue (no `Self.ArrayType` projection → no recursion).
    def execute(self, batch: Batch) raises -> ColumnarValue:
        comptime if Self.Shape == 0:  # scalar → evaluate once
            return ColumnarValue.scalar(self.core[1](batch, 0)[0])
        else:  # columnar → one fused W-blocked pass, no temporaries
            var n = batch.num_rows()
            var out = List[Int64](capacity=n)
            var i = 0
            while i + 4 <= n:
                var v = self.core[4](batch, i)
                for k in range(4):
                    out.append(v[k])
                i += 4
            while i < n:
                out.append(self.core[1](batch, i)[0])
                i += 1
            return ColumnarValue.columnar(out^)


# ===========================================================================
# Fusable nodes — implement core, fuse for free
# ===========================================================================
@fieldwise_init
struct Col(Fusable):
    comptime Shape = 1
    var col: Int

    @always_inline
    def core[W: Int](self, batch: Batch, idx: Int) -> SIMD[i64, W]:
        return load[W](batch.cols[self.col], idx)


@fieldwise_init
struct Lit(Fusable):
    comptime Shape = 0
    var v: Int64

    @always_inline
    def core[W: Int](self, batch: Batch, idx: Int) -> SIMD[i64, W]:
        return SIMD[i64, W](self.v)  # splat


@fieldwise_init
struct Add[L: Fusable, R: Fusable](Fusable):
    comptime Shape = max(Self.L.Shape, Self.R.Shape)
    var l: Self.L
    var r: Self.R

    @always_inline
    def core[W: Int](self, batch: Batch, idx: Int) -> SIMD[i64, W]:
        return AddKernel.core[W](
            self.l.core[W](batch, idx), self.r.core[W](batch, idx)
        )


# ===========================================================================
# Cross-row nodes — implement execute (materialize). No core, no stored state.
# ===========================================================================
@fieldwise_init
struct Sum[A: Value](Value):
    comptime Shape = 0
    var a: Self.A

    def execute(self, batch: Batch) raises -> ColumnarValue:
        var col = self.a.execute(batch).into_array(batch.num_rows())
        var s = Int64(0)
        for i in range(len(col)):
            s += col[i]
        return ColumnarValue.scalar(s)


struct WindowFunction[Func: WindowKernel, A: Value](Value):
    comptime Shape = 1
    var arg: Self.A
    var spec: WindowSpec

    def __init__(out self, var arg: Self.A, var spec: WindowSpec):
        self.arg = arg^
        self.spec = spec^

    def execute(self, batch: Batch) raises -> ColumnarValue:
        var n = batch.num_rows()
        var v = self.arg.execute(batch).into_array(n)
        for i in range(len(self.spec.partition_by)):  # prove erased keys evaluate
            var key = self.spec.partition_by[i].execute(batch)
            _ = key.num_rows()
        return ColumnarValue.columnar(Self.Func.evaluate_all(v))


# Mixed arithmetic with a cross-row operand — eager, one kernel `apply`.
# (Real version is parameterized by the kernel; hardcoded to add here.)
@fieldwise_init
struct MatBinary[L: Value, R: Value](Value):
    comptime Shape = max(Self.L.Shape, Self.R.Shape)
    var l: Self.L
    var r: Self.R

    def execute(self, batch: Batch) raises -> ColumnarValue:
        return AddKernel.apply(self.l.execute(batch), self.r.execute(batch))


# ===========================================================================
# AnyValue — erase any Value, execute() -> ColumnarValue
# ===========================================================================
struct AnyValue(Copyable, Movable):
    var _boxed: ArcPointer[NoneType]
    var _execute: def (
        ArcPointer[NoneType], Batch
    ) thin raises -> ColumnarValue

    @staticmethod
    def _exec_tramp[
        V: Value
    ](ptr: ArcPointer[NoneType], batch: Batch) raises -> ColumnarValue:
        return rebind[ArcPointer[V]](ptr)[].execute(batch)

    @implicit
    def __init__[V: Value](out self, value: V):
        var ptr = ArcPointer[V](value.copy())
        self._boxed = rebind[ArcPointer[NoneType]](ptr^)
        self._execute = Self._exec_tramp[V]

    def execute(self, batch: Batch) raises -> ColumnarValue:
        return self._execute(self._boxed, batch)


@fieldwise_init
struct FrameBound(Copyable, ImplicitlyCopyable, Movable):
    var kind: UInt8
    var offset: Int64


struct WindowSpec(Copyable, Movable):
    var start: FrameBound
    var end: FrameBound
    var partition_by: List[AnyValue]
    var order_by: List[AnyValue]

    def __init__(
        out self,
        start: FrameBound,
        end: FrameBound,
        var partition_by: List[AnyValue],
        var order_by: List[AnyValue],
    ):
        self.start = start
        self.end = end
        self.partition_by = partition_by^
        self.order_by = order_by^


# ===========================================================================
# Dynamic driver — runtime tree, eager, returns ColumnarValue, shares AddKernel
# ===========================================================================
struct DynExpr(Copyable, ImplicitlyDeletable, Movable):
    var tag: UInt8
    var col: Int
    var lit: Int64
    var kids: List[ArcPointer[DynExpr]]

    def __init__(
        out self, tag: UInt8, col: Int, lit: Int64, var kids: List[ArcPointer[DynExpr]]
    ):
        self.tag = tag
        self.col = col
        self.lit = lit
        self.kids = kids^

    @staticmethod
    def column(c: Int) -> Self:
        return Self(0, c, 0, List[ArcPointer[DynExpr]]())

    @staticmethod
    def add(var a: DynExpr, var b: DynExpr) -> Self:
        var k = List[ArcPointer[DynExpr]]()
        k.append(ArcPointer[DynExpr](a^))
        k.append(ArcPointer[DynExpr](b^))
        return Self(2, 0, 0, k^)

    @staticmethod
    def sum(var a: DynExpr) -> Self:
        var k = List[ArcPointer[DynExpr]]()
        k.append(ArcPointer[DynExpr](a^))
        return Self(3, 0, 0, k^)

    def eval(self, batch: Batch) raises -> ColumnarValue:
        var n = batch.num_rows()
        if self.tag == 0:
            return ColumnarValue.columnar(batch.cols[self.col].copy())
        if self.tag == 2:
            return AddKernel.apply(
                self.kids[0][].eval(batch), self.kids[1][].eval(batch)
            )
        var a = self.kids[0][].eval(batch).into_array(n)  # sum
        var s = Int64(0)
        for i in range(len(a)):
            s += a[i]
        return ColumnarValue.scalar(s)


# ===========================================================================
def _spec() raises -> WindowSpec:
    var pb: List[AnyValue] = [AnyValue(Col(1))]
    var ob = List[AnyValue]()
    return WindowSpec(FrameBound(0, 0), FrameBound(2, 0), pb^, ob^)


def _fmt(xs: List[Int64]) -> String:
    var s = String("[")
    for i in range(len(xs)):
        if i:
            s += ", "
        s += String(xs[i])
    s += "]"
    return s


def _show(name: String, cv: ColumnarValue) raises:
    if cv.is_scalar():
        print(name, "= scalar", cv.scalar_value())
    else:
        print(name, "= columnar", _fmt(cv.into_array(cv.num_rows())))


def main() raises:
    var c0: List[Int64] = [3, 1, 2, 4]
    var c1: List[Int64] = [0, 0, 1, 1]
    var cols: List[List[Int64]] = [c0^, c1^]
    var batch = Batch(cols^)

    # --- fully lane → fused (Add over Fusable children) ---
    _show("col + 10          ", Add(Col(0), Lit(10)).execute(batch))

    # --- cross-row operand → MatBinary (eager execute) ---
    _show("col + sum(col)    ", MatBinary(Col(0), Sum(Col(0))).execute(batch))
    _show("sum(col) + 5      ", MatBinary(Sum(Col(0)), Lit(5)).execute(batch))  # scalar

    # --- TWO cross-row nodes, no addressing needed (each just returns a CV) ---
    _show(
        "sum(col)+rank(col) ",
        MatBinary(Sum(Col(0)), WindowFunction[RowNumberKernel](Col(0), _spec())).execute(batch),
    )
    _show(
        "rank(col) + 1      ",
        MatBinary(WindowFunction[RowNumberKernel](Col(0), _spec()), Lit(1)).execute(batch),
    )

    # --- AnyValue erasure ---
    var boxed: AnyValue = Add(Col(0), Lit(10))
    _show("erased col + 10   ", boxed.execute(batch))

    # --- dynamic driver, shared AddKernel ---
    var dyn = DynExpr.add(DynExpr.column(0), DynExpr.sum(DynExpr.column(0)))
    _show("dynamic col+sum   ", dyn.eval(batch))
