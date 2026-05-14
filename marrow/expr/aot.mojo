"""AOT query compilation via compile-time plan types.

Encodes the query plan as a Mojo type using compile-time parameters so the
compiler can specialize every operator for that exact plan, eliminating all
vtable dispatch and intermediate array allocations.

The two execution modes share the same builder API — only the binding keyword
differs:

    # AOT: compiler specializes run_plan for this exact plan type
    alias plan = ct_scan[orders_schema].filter(
        col[0, Int64Type]() + lit_i[1]() > lit_i[0]()
    )
    var result = run_plan(plan.bind(source_batch))

    # Runtime bridge: same expression tree, falls back to interpreter
    var rt_plan = plan.to_any_relation()
    var result   = execute(rt_plan, ctx)
"""

from marrow.arrays import AnyArray
from marrow.dtypes import DataType, Int64Type, Float64Type, AnyDataType, Field
from marrow.schema import Schema
from marrow.tabular import RecordBatch
from marrow.kernels.arithmetic import add, subtract, multiply, divide
from marrow.kernels.compare import equal, not_equal, less, less_equal, greater, greater_equal
from marrow.kernels.boolean import and_, or_, not_
from marrow.kernels.filter import filter as filter_batch
from marrow.kernels.execution import ExecutionContext
from marrow.expr.values import (
    ADD, SUB, MUL, DIV, EQ, NE, LT, LE, GT, GE, AND, OR, NEG, NOT,
    AnyValue,
    Column,
    Binary as RtBinary,
    Literal as RtLiteral,
    Unary as RtUnary,
)
from marrow.expr.relations import (
    AnyRelation,
    FILTER_NODE, JOIN_NODE, IN_MEMORY_TABLE_NODE,
)

comptime MORSEL_SIZE: Int = 65_536


# ─────────────────────────────────────────────────────────────────────────────
# CtExpr — compile-time scalar expression
# ─────────────────────────────────────────────────────────────────────────────


trait CtExpr(ImplicitlyCopyable, TrivialRegisterPassable):
    """Zero-size compile-time scalar expression.

    All information lives in the type parameters. `eval()` is `@always_inline`
    so the compiler sees through the entire expression tree in one inlining
    pass and emits a single fused loop with no intermediate arrays.
    """

    @always_inline
    fn eval(self, batch: RecordBatch) raises -> AnyArray:
        """Evaluate this expression against a batch. Fully inlined by compiler."""
        ...

    fn to_runtime(self) -> AnyValue:
        """Bridge to the runtime interpreter path."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# ColRef — compile-time column reference
# ─────────────────────────────────────────────────────────────────────────────


struct ColRef[idx: Int, T: DataType](CtExpr):
    """Reference to column `idx` of the input batch with compile-time type `T`."""

    @always_inline
    fn eval(self, batch: RecordBatch) raises -> AnyArray:
        return batch.columns[idx].copy()

    fn to_runtime(self) -> AnyValue:
        return Column(index=idx, name="", dtype_=Optional(T().to_any()))

    # ── operator overloads — each returns the fully-typed Binary/Unary ──

    fn __add__[R: CtExpr](self, rhs: R) -> Binary[ADD, Self, R]:
        return {}

    fn __sub__[R: CtExpr](self, rhs: R) -> Binary[SUB, Self, R]:
        return {}

    fn __mul__[R: CtExpr](self, rhs: R) -> Binary[MUL, Self, R]:
        return {}

    fn __truediv__[R: CtExpr](self, rhs: R) -> Binary[DIV, Self, R]:
        return {}

    fn __eq__[R: CtExpr](self, rhs: R) -> Binary[EQ, Self, R]:
        return {}

    fn __ne__[R: CtExpr](self, rhs: R) -> Binary[NE, Self, R]:
        return {}

    fn __lt__[R: CtExpr](self, rhs: R) -> Binary[LT, Self, R]:
        return {}

    fn __le__[R: CtExpr](self, rhs: R) -> Binary[LE, Self, R]:
        return {}

    fn __gt__[R: CtExpr](self, rhs: R) -> Binary[GT, Self, R]:
        return {}

    fn __ge__[R: CtExpr](self, rhs: R) -> Binary[GE, Self, R]:
        return {}

    fn __and__[R: CtExpr](self, rhs: R) -> Binary[AND, Self, R]:
        return {}

    fn __or__[R: CtExpr](self, rhs: R) -> Binary[OR, Self, R]:
        return {}

    fn __neg__(self) -> Unary[NEG, Self]:
        return {}

    fn __invert__(self) -> Unary[NOT, Self]:
        return {}

    fn cast[To: DataType](self) -> Cast[Self, To]:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# IntLit / FloatLit — compile-time scalar literals
#
# Separate types because Mojo parameter types must be homogeneous — a single
# `Literal[val: ???]` that covers both Int and Float64 isn't expressible
# without an intermediate union type.
# ─────────────────────────────────────────────────────────────────────────────


struct IntLit[val: Int](CtExpr):
    """Compile-time integer literal. Broadcasts to batch length on eval."""

    @always_inline
    fn eval(self, batch: RecordBatch) raises -> AnyArray:
        from marrow.builders import PrimitiveBuilder
        var b = PrimitiveBuilder[Int64Type]()
        for _ in range(batch.num_rows()):
            b.append_value(Int64(val))
        return b.finish().to_any()

    fn to_runtime(self) -> AnyValue:
        from marrow.builders import PrimitiveBuilder
        var b = PrimitiveBuilder[Int64Type]()
        b.append_value(Int64(val))
        return RtLiteral(value=b.finish().to_any())

    fn __add__[R: CtExpr](self, rhs: R) -> Binary[ADD, Self, R]:
        return {}
    fn __sub__[R: CtExpr](self, rhs: R) -> Binary[SUB, Self, R]:
        return {}
    fn __eq__[R: CtExpr](self, rhs: R) -> Binary[EQ, Self, R]:
        return {}
    fn __ne__[R: CtExpr](self, rhs: R) -> Binary[NE, Self, R]:
        return {}
    fn __lt__[R: CtExpr](self, rhs: R) -> Binary[LT, Self, R]:
        return {}
    fn __le__[R: CtExpr](self, rhs: R) -> Binary[LE, Self, R]:
        return {}
    fn __gt__[R: CtExpr](self, rhs: R) -> Binary[GT, Self, R]:
        return {}
    fn __ge__[R: CtExpr](self, rhs: R) -> Binary[GE, Self, R]:
        return {}
    fn __and__[R: CtExpr](self, rhs: R) -> Binary[AND, Self, R]:
        return {}
    fn __or__[R: CtExpr](self, rhs: R) -> Binary[OR, Self, R]:
        return {}
    fn __neg__(self) -> Unary[NEG, Self]:
        return {}


struct FloatLit[val: Float64](CtExpr):
    """Compile-time float literal."""

    @always_inline
    fn eval(self, batch: RecordBatch) raises -> AnyArray:
        from marrow.builders import PrimitiveBuilder
        var b = PrimitiveBuilder[Float64Type]()
        for _ in range(batch.num_rows()):
            b.append_value(val)
        return b.finish().to_any()

    fn to_runtime(self) -> AnyValue:
        from marrow.builders import PrimitiveBuilder
        var b = PrimitiveBuilder[Float64Type]()
        b.append_value(val)
        return RtLiteral(value=b.finish().to_any())

    fn __add__[R: CtExpr](self, rhs: R) -> Binary[ADD, Self, R]:
        return {}
    fn __sub__[R: CtExpr](self, rhs: R) -> Binary[SUB, Self, R]:
        return {}
    fn __eq__[R: CtExpr](self, rhs: R) -> Binary[EQ, Self, R]:
        return {}
    fn __ne__[R: CtExpr](self, rhs: R) -> Binary[NE, Self, R]:
        return {}
    fn __lt__[R: CtExpr](self, rhs: R) -> Binary[LT, Self, R]:
        return {}
    fn __le__[R: CtExpr](self, rhs: R) -> Binary[LE, Self, R]:
        return {}
    fn __gt__[R: CtExpr](self, rhs: R) -> Binary[GT, Self, R]:
        return {}
    fn __ge__[R: CtExpr](self, rhs: R) -> Binary[GE, Self, R]:
        return {}
    fn __and__[R: CtExpr](self, rhs: R) -> Binary[AND, Self, R]:
        return {}
    fn __or__[R: CtExpr](self, rhs: R) -> Binary[OR, Self, R]:
        return {}
    fn __neg__(self) -> Unary[NEG, Self]:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Binary — compile-time binary expression
# ─────────────────────────────────────────────────────────────────────────────


struct Binary[op: UInt8, L: CtExpr, R: CtExpr](CtExpr):
    """Binary expression with compile-time operator.

    `if op == ADD:` is a comptime branch — the compiler eliminates all branches
    that don't match `op` for this specialization, emitting only the one
    matching kernel call.
    """

    @always_inline
    fn eval(self, batch: RecordBatch) raises -> AnyArray:
        var l = L().eval(batch)
        var r = R().eval(batch)
        # All dead branches eliminated at compile time because op is a
        # comptime UInt8 parameter.
        if op == ADD:
            return add(l, r)
        elif op == SUB:
            return subtract(l, r)
        elif op == MUL:
            return multiply(l, r)
        elif op == DIV:
            return divide(l, r)
        elif op == EQ:
            return equal(l, r)
        elif op == NE:
            return not_equal(l, r)
        elif op == LT:
            return less(l, r)
        elif op == LE:
            return less_equal(l, r)
        elif op == GT:
            return greater(l, r)
        elif op == GE:
            return greater_equal(l, r)
        elif op == AND:
            return and_(l, r)
        elif op == OR:
            return or_(l, r)
        else:
            return l  # unreachable

    fn to_runtime(self) -> AnyValue:
        return RtBinary(op=op, left=L().to_runtime(), right=R().to_runtime())

    fn __add__[R2: CtExpr](self, rhs: R2) -> Binary[ADD, Self, R2]:
        return {}
    fn __sub__[R2: CtExpr](self, rhs: R2) -> Binary[SUB, Self, R2]:
        return {}
    fn __eq__[R2: CtExpr](self, rhs: R2) -> Binary[EQ, Self, R2]:
        return {}
    fn __ne__[R2: CtExpr](self, rhs: R2) -> Binary[NE, Self, R2]:
        return {}
    fn __lt__[R2: CtExpr](self, rhs: R2) -> Binary[LT, Self, R2]:
        return {}
    fn __le__[R2: CtExpr](self, rhs: R2) -> Binary[LE, Self, R2]:
        return {}
    fn __gt__[R2: CtExpr](self, rhs: R2) -> Binary[GT, Self, R2]:
        return {}
    fn __ge__[R2: CtExpr](self, rhs: R2) -> Binary[GE, Self, R2]:
        return {}
    fn __and__[R2: CtExpr](self, rhs: R2) -> Binary[AND, Self, R2]:
        return {}
    fn __or__[R2: CtExpr](self, rhs: R2) -> Binary[OR, Self, R2]:
        return {}
    fn __neg__(self) -> Unary[NEG, Self]:
        return {}
    fn __invert__(self) -> Unary[NOT, Self]:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Unary — compile-time unary expression
# ─────────────────────────────────────────────────────────────────────────────


struct Unary[op: UInt8, C: CtExpr](CtExpr):
    @always_inline
    fn eval(self, batch: RecordBatch) raises -> AnyArray:
        var c = C().eval(batch)
        if op == NEG:
            from marrow.kernels.arithmetic import negate
            return negate(c)
        elif op == NOT:
            return not_(c)
        else:
            return c  # unreachable

    fn to_runtime(self) -> AnyValue:
        return RtUnary(op=op, child=C().to_runtime())

    fn __add__[R: CtExpr](self, rhs: R) -> Binary[ADD, Self, R]:
        return {}
    fn __gt__[R: CtExpr](self, rhs: R) -> Binary[GT, Self, R]:
        return {}
    fn __lt__[R: CtExpr](self, rhs: R) -> Binary[LT, Self, R]:
        return {}
    fn __and__[R: CtExpr](self, rhs: R) -> Binary[AND, Self, R]:
        return {}
    fn __or__[R: CtExpr](self, rhs: R) -> Binary[OR, Self, R]:
        return {}
    fn __neg__(self) -> Unary[NEG, Self]:
        return {}
    fn __invert__(self) -> Unary[NOT, Self]:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Cast — compile-time explicit type cast
# ─────────────────────────────────────────────────────────────────────────────


struct Cast[C: CtExpr, To: DataType](CtExpr):
    @always_inline
    fn eval(self, batch: RecordBatch) raises -> AnyArray:
        from marrow.kernels.cast import cast
        return cast(C().eval(batch), To().to_any())

    fn to_runtime(self) -> AnyValue:
        from marrow.expr.values import Cast as RtCast
        return RtCast(child=C().to_runtime(), to=To().to_any())

    fn __add__[R: CtExpr](self, rhs: R) -> Binary[ADD, Self, R]:
        return {}
    fn __gt__[R: CtExpr](self, rhs: R) -> Binary[GT, Self, R]:
        return {}
    fn __and__[R: CtExpr](self, rhs: R) -> Binary[AND, Self, R]:
        return {}
    fn __neg__(self) -> Unary[NEG, Self]:
        return {}
    fn __invert__(self) -> Unary[NOT, Self]:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# CtRelation — compile-time relational plan node
# ─────────────────────────────────────────────────────────────────────────────


trait CtRelation(Movable):
    """Compile-time relational plan node.

    `alias schema: Schema` is always known at compile time regardless of
    whether the computation is AOT or runtime-bridged. `pull()` is the
    morsel-pull interface used by `run_plan`.
    """

    alias schema: Schema

    fn pull(mut self) raises -> Optional[RecordBatch]:
        """Pull the next morsel. Returns None when exhausted."""
        ...

    fn to_any_relation(self) -> AnyRelation:
        """Bridge to the runtime interpreter path."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# CtScan — leaf: reads from a RecordBatch in MORSEL_SIZE chunks
# ─────────────────────────────────────────────────────────────────────────────


struct CtScan[s: Schema](CtRelation):
    alias schema: Schema = s
    var source: RecordBatch
    var offset: Int

    def __init__(out self, source: RecordBatch):
        self.source = source
        self.offset = 0

    fn pull(mut self) raises -> Optional[RecordBatch]:
        if self.offset >= self.source.num_rows():
            return None
        var morsel = self.source.slice(self.offset, MORSEL_SIZE)
        self.offset += morsel.num_rows()
        return morsel

    fn to_any_relation(self) -> AnyRelation:
        from marrow.expr.relations import InMemoryTable
        return AnyRelation(InMemoryTable(batch=self.source))

    # ── plan-builder methods ──────────────────────────────────────────────

    fn filter[P: CtExpr](owned self, pred: P) -> CtFilter[Self, P]:
        return CtFilter[Self, P](child=self^)

    fn join[
        Right: CtRelation, LK: CtExpr, RK: CtExpr
    ](owned self, right: Right, left_on: LK, right_on: RK) -> CtHashJoin[Self, Right, LK, RK]:
        return CtHashJoin[Self, Right, LK, RK](left=self^, right=right)

    fn limit(owned self, comptime n: Int) -> CtLimit[Self, n]:
        return CtLimit[Self, n](child=self^)


# ─────────────────────────────────────────────────────────────────────────────
# CtFilter — filter rows by a compile-time predicate
# ─────────────────────────────────────────────────────────────────────────────


struct CtFilter[Child: CtRelation, Pred: CtExpr](CtRelation):
    alias schema: Schema = Child.schema  # filter never changes the schema
    var child: Child

    def __init__(out self, owned child: Child):
        self.child = child^

    fn pull(mut self) raises -> Optional[RecordBatch]:
        while True:
            var maybe = self.child.pull()
            if not maybe:
                return None
            var batch = maybe.value()
            # Pred.eval() is @always_inline — inlined through the full
            # expression tree before this loop body reaches LLVM.
            var mask = Pred().eval(batch)
            var filtered = _filter_batch(batch, mask)
            if filtered.num_rows() > 0:
                return filtered

    fn to_any_relation(self) -> AnyRelation:
        from marrow.expr.relations import Filter
        return AnyRelation(Filter(
            input=self.child.to_any_relation(),
            predicate=Pred().to_runtime(),
            schema_=Child.schema,
        ))

    fn filter[P: CtExpr](owned self, pred: P) -> CtFilter[Self, P]:
        return CtFilter[Self, P](child=self^)

    fn join[
        Right: CtRelation, LK: CtExpr, RK: CtExpr
    ](owned self, right: Right, left_on: LK, right_on: RK) -> CtHashJoin[Self, Right, LK, RK]:
        return CtHashJoin[Self, Right, LK, RK](left=self^, right=right)

    fn limit(owned self, comptime n: Int) -> CtLimit[Self, n]:
        return CtLimit[Self, n](child=self^)


# ─────────────────────────────────────────────────────────────────────────────
# CtHashJoin — hash join with compile-time key expressions
# ─────────────────────────────────────────────────────────────────────────────


fn _concat_schemas(s1: Schema, s2: Schema) -> Schema:
    """Compile-time schema concatenation for join output."""
    var result = Schema()
    for field in s1.fields:
        result.append(field.copy())
    for field in s2.fields:
        result.append(field.copy())
    return result^


struct CtHashJoin[
    Left: CtRelation,
    Right: CtRelation,
    LeftKey: CtExpr,
    RightKey: CtExpr,
](CtRelation):
    # Schema is the concatenation of left and right, computed at compile time.
    alias schema: Schema = _concat_schemas(Left.schema, Right.schema)
    var left: Left
    var right: Right
    var _ht: Optional[_HashTable]  # built lazily on first pull

    def __init__(out self, owned left: Left, owned right: Right):
        self.left = left^
        self.right = right^
        self._ht = None

    fn pull(mut self) raises -> Optional[RecordBatch]:
        if not self._ht:
            self._ht = self._build()
        return self._probe_next()

    fn _build(mut self) raises -> _HashTable:
        var ht = _HashTable()
        while True:
            var maybe = self.left.pull()
            if not maybe:
                break
            var batch = maybe.value()
            # LeftKey.eval() is @always_inline — no vtable, no dispatch.
            var key_col = LeftKey().eval(batch)
            ht.insert(key_col, batch)
        return ht^

    fn _probe_next(mut self) raises -> Optional[RecordBatch]:
        # TODO: drive probe side morsel-by-morsel with RightKey().eval()
        ...

    fn to_any_relation(self) -> AnyRelation:
        from marrow.expr.relations import Join, JOIN_INNER, JOIN_ALGO_HASH
        return AnyRelation(Join(
            left=self.left.to_any_relation(),
            right=self.right.to_any_relation(),
            left_keys=[LeftKey().to_runtime()],
            right_keys=[RightKey().to_runtime()],
            kind=JOIN_INNER,
            algorithm=JOIN_ALGO_HASH,
        ))

    fn filter[P: CtExpr](owned self, pred: P) -> CtFilter[Self, P]:
        return CtFilter[Self, P](child=self^)

    fn limit(owned self, comptime n: Int) -> CtLimit[Self, n]:
        return CtLimit[Self, n](child=self^)


# ─────────────────────────────────────────────────────────────────────────────
# CtLimit — compile-time row count cap
# ─────────────────────────────────────────────────────────────────────────────


struct CtLimit[Child: CtRelation, n: Int](CtRelation):
    alias schema: Schema = Child.schema
    var child: Child
    var _emitted: Int

    def __init__(out self, owned child: Child):
        self.child = child^
        self._emitted = 0

    fn pull(mut self) raises -> Optional[RecordBatch]:
        if self._emitted >= n:
            return None
        var maybe = self.child.pull()
        if not maybe:
            return None
        var batch = maybe.value()
        var remaining = n - self._emitted
        if batch.num_rows() > remaining:
            batch = batch.slice(0, remaining)
        self._emitted += batch.num_rows()
        return batch

    fn to_any_relation(self) -> AnyRelation:
        from marrow.expr.relations import Limit
        return AnyRelation(Limit(input=self.child.to_any_relation(), count=n))

    fn filter[P: CtExpr](owned self, pred: P) -> CtFilter[Self, P]:
        return CtFilter[Self, P](child=self^)

    fn limit(owned self, comptime m: Int) -> CtLimit[Self, m]:
        return CtLimit[Self, m](child=self^)


# ─────────────────────────────────────────────────────────────────────────────
# run_plan — AOT entry point
#
# `P` is specialized at compile time. `plan.pull()` inlines the entire
# operator tree through P's type — no vtable, no runtime dispatch.
# ─────────────────────────────────────────────────────────────────────────────


def run_plan[P: CtRelation](owned plan: P) raises -> RecordBatch:
    """Execute a compile-time plan, collecting all output into one RecordBatch."""
    var out = RecordBatch(schema=P.schema, columns=[])
    while True:
        var maybe = plan.pull()
        if not maybe:
            break
        out.append(maybe.value())
    return out^


# ─────────────────────────────────────────────────────────────────────────────
# Factory helpers — ergonomic plan / expression construction
# ─────────────────────────────────────────────────────────────────────────────


fn col[idx: Int, T: DataType]() -> ColRef[idx, T]:
    """Column reference by index with explicit type."""
    return {}


fn lit_i[val: Int]() -> IntLit[val]:
    """Integer literal."""
    return {}


fn lit_f[val: Float64]() -> FloatLit[val]:
    """Float literal."""
    return {}


fn ct_scan[s: Schema](source: RecordBatch) -> CtScan[s]:
    """Scan over a RecordBatch with a compile-time known schema."""
    return CtScan[s](source=source)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


fn _filter_batch(batch: RecordBatch, mask: AnyArray) raises -> RecordBatch:
    """Apply a boolean mask to every column of a batch."""
    var cols = List[AnyArray]()
    for col_arr in batch.columns:
        cols.append(filter_batch(col_arr.copy(), mask))
    return RecordBatch(schema=batch.schema, columns=cols^)


struct _HashTable:
    """Minimal hash table for join build phase. Placeholder — real
    implementation would use marrow.kernels.hashtable."""

    var _keys: List[AnyArray]
    var _batches: List[RecordBatch]

    def __init__(out self):
        self._keys = List[AnyArray]()
        self._batches = List[RecordBatch]()

    def insert(mut self, key: AnyArray, batch: RecordBatch):
        self._keys.append(key)
        self._batches.append(batch)
