"""The vocabulary a golden case body may use, on the Python side.

`helpers.mojo` is the same list for the Mojo lane, and `runner.MOJO_HEADER`
imports it. Between them they are the convergence contract: a name a case can
write is a name both lanes answer to.

**Most of what is here is a shim, and that is the point.** A shim exists
wherever the two lanes do not yet agree on a spelling — `Upper(x)` standing in
for `x.upper()`, `NumericCast[Float64Type](x)` for `x.cast(float64)`. Marrow's
*public* Python API is deliberately not grown to speak the fused lane's
internal node vocabulary, so the bridge lives here, in the corpus, where it is
countable.

The aggregates used to be the worst of it —
`AggExpr.of[NumericAgg[SumKernel, Int64Type]](x).alias("total")` — and needed
nine shims. It turned out marrow already had `col("v", int64).sum()`, on
`NumericValue`, documented in `Relation.aggregate` and used throughout
`marrow/expr/tests`; the corpus was simply spelling it the long way. That is
the shape of the remaining work: check whether the nicer spelling already
exists before designing one.

`SHIMS` is that count. It is the convergence metric: every name in it is a
place the two lanes still disagree, and the goal is an empty set. Deleting a
shim means the real APIs converged, which is the design target this corpus
exists to hold marrow to.
"""

import pyarrow as pa

import marrow
from marrow import Column, col, count_star, if_else
from marrow import lit as _lit

import runner

# Names that are *not* real marrow API — the outstanding convergence debt.
# Keep this in sync with what is defined below; `test_cases.py` reports it.
SHIMS = {
    # dtype spellings. Mojo has `int64` as a dtype *value* and `Int64Type` as
    # the type a fused node is parameterised on; Python has one constructor,
    # `marrow.int64()`, and so must call it for both.
    "int64",
    "int32",
    "float64",
    "string",
    "bool_",
    "Int64Type",
    "Int32Type",
    "Float64Type",
    "StringType",
    "BoolType",
    # fused string nodes vs. Python methods
    "Upper",
    "Lower",
    "Strip",
    "StringLength",
    "StartsWith",
    "EndsWith",
    "Like",
    "ILike",
    # `lit("h%")` as a string kernel's operand: the fused lane takes a value
    # node, Python takes a plain pattern (as PyArrow does).
    "lit",
    # fused null / conditional nodes vs. Python methods
    "IsNull",
    "NotNull",
    "CaseWhen",
    "Coalesce",
    "FillNull",
    # fused cast nodes vs. `.cast(type, safe=False)`
    "NumericCast",
    "NumToString",
    "StringToNum",
    "BoolToNum",
    "NumToBool",
    # join kinds: Mojo constants vs. Python strings
    "JOIN_INNER",
    "JOIN_LEFT",
    "JOIN_RIGHT",
    "JOIN_FULL",
    "JOIN_SEMI",
    "JOIN_ANTI",
    "JOIN_ALL",
    # relational verbs whose shapes differ (`sort`/`aggregate`/`join`) —
    # carried by the `_Relation` adapter rather than by `LazyTable` itself.
    "_Relation.sort",
    "_Relation.aggregate",
    "_Relation.join",
}


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
# Mojo distinguishes the dtype *value* (`int64`) from the dtype *type*
# (`Int64Type`); a fused node is parameterised on the latter. Python has one
# spelling for both.

int64 = marrow.int64()
int32 = marrow.int32()
float64 = marrow.float64()
string = marrow.string()
bool_ = marrow.bool_()

Int64Type = int64
Int32Type = int32
Float64Type = float64
StringType = string
BoolType = bool_


# ---------------------------------------------------------------------------
# Fused value nodes -> Python expression methods
# ---------------------------------------------------------------------------


class _Literal(Column):
    """A literal that remembers the Python value it was built from.

    The fused lane's string kernels take a value node — `Like(s, lit("h%"))`.
    Python's take a plain pattern, as PyArrow's `match_like` does, and handing
    them a `Column` matches nothing rather than raising. `Column` defines
    `__slots__`, so the raw value cannot be tagged onto one; a subclass with
    its own slot can carry it.
    """

    __slots__ = ("value",)


def lit(value, type=None):
    out = _Literal.wrap(_lit(value, type).unwrap())
    out.value = value
    return out


def _pattern(value):
    """The raw pattern behind a `lit(...)`, or the value itself."""
    return value.value if isinstance(value, _Literal) else value


def Upper(value):
    return value.upper()


def Lower(value):
    return value.lower()


def Strip(value):
    return value.strip()


def StringLength(value):
    return value.length()


def StartsWith(value, prefix):
    return value.startswith(_pattern(prefix))


def EndsWith(value, suffix):
    return value.endswith(_pattern(suffix))


def Like(value, pattern):
    return value.like(_pattern(pattern))


def ILike(value, pattern):
    return value.ilike(_pattern(pattern))


def IsNull(value):
    return value.is_null()


def NotNull(value):
    return value.is_valid()


def CaseWhen(condition, then, otherwise):
    return if_else(condition, then, otherwise)


def Coalesce(value, other):
    return value.coalesce(other)


def FillNull(value, other):
    return value.fill_null(other)


# ---------------------------------------------------------------------------
# Casts
# ---------------------------------------------------------------------------
# Every case casts with `safe=False`: a lossy conversion under the default
# `safe=True` raises, which is a different question from what a SQL CAST does.


class _Cast:
    """`NumericCast[Float64Type](x)` -> `x.cast(float64, safe=False)`."""

    def __class_getitem__(cls, target):
        return lambda value: value.cast(target, safe=False)


class NumericCast(_Cast):
    pass


class NumToString(_Cast):
    pass


class StringToNum(_Cast):
    pass


class BoolToNum(_Cast):
    pass


def NumToBool(value):
    """Unparameterised in the fused lane — the target is always `bool`."""
    return value.cast(bool_, safe=False)


# ---------------------------------------------------------------------------
# Joins
# ---------------------------------------------------------------------------

JOIN_INNER = "inner"
JOIN_LEFT = "left"
JOIN_RIGHT = "right"
JOIN_FULL = "full"
JOIN_SEMI = "semi"
JOIN_ANTI = "anti"
JOIN_ALL = "all"


# ---------------------------------------------------------------------------
# The relation surface
# ---------------------------------------------------------------------------


class _Relation:
    """A `LazyTable` wearing the Mojo plan API's verb shapes.

    `select`, `filter`, `project`, `with_columns` and `limit` pass straight
    through — those already agree. `sort`, `aggregate` and `join` do not, so
    they are adapted here rather than by growing `LazyTable` a second spelling
    of each.
    """

    def __init__(self, lazy):
        self._lazy = lazy

    def select(self, *names):
        return _Relation(self._lazy.select(*names))

    def filter(self, predicate):
        return _Relation(self._lazy.filter(predicate))

    def project(self, names, values):
        return _Relation(self._lazy.project(names, values))

    def with_columns(self, names, values):
        return _Relation(self._lazy.with_columns(names, values))

    def limit(self, length, offset=0):
        return _Relation(self._lazy.limit(length, offset))

    def sort(self, keys, ascending, nulls_first=True):
        """Parallel key and direction lists, as the plan node takes them.

        `LazyTable.order_by` spells this as `("k", "descending")` pairs. The
        binding underneath already takes the parallel form, so this reaches
        past the Python sugar rather than reconstructing it.
        """
        return _Relation(
            marrow.expr.LazyTable.wrap(
                self._lazy.unwrap().sort(
                    [k.unwrap() for k in keys], list(ascending), nulls_first, True
                )
            )
        )

    def aggregate(self, aggs, keys=()):
        """`keys` is optional, matching the Mojo overload.

        No-GROUP-BY is `t.aggregate(aggs=[col("v", int64).sum()])` in both
        lanes — polars and ibis both let the key list be absent, and an empty
        one carried no information.
        """
        return _Relation(self._lazy.aggregate(keys, *aggs))

    def join(self, other, left_on, right_on, how, strictness):
        return _Relation(
            self._lazy.join(
                other._lazy,
                left_on=left_on,
                right_on=right_on,
                how=how,
                strictness=strictness,
            )
        )

    def to_pyarrow(self, num_threads=0):
        return self._lazy.to_pyarrow(num_threads=num_threads)


def table(name):
    """A fixture as an in-memory source — never a file scan.

    What is under test is the engine, so the source is a memtable in every
    lane; Parquet and IPC keep their own suites.
    """
    batch = marrow.read_ipc_file(str(runner.fixture_path(name)))[0]
    return _Relation(marrow.memtable(batch, morsel_size=runner.MORSEL_SIZE))


def check(name, plan):
    """Run the plan and hold it to the shared expectation."""
    expected = _expectations()[name]
    actual = pa.table(plan.to_pyarrow(num_threads=runner.NUM_THREADS))
    if actual.equals(expected):
        return
    raise AssertionError(
        f"{name} does not match its expectation\n\n"
        f"--- expected (duckdb) ---\n{expected}\n"
        f"--- actual (marrow) ---\n{actual}\n"
    )


_EXPECTED = None


def _expectations():
    global _EXPECTED
    if _EXPECTED is None:
        _EXPECTED = {case.name: case.expected for case in runner.load_cases()}
    return _EXPECTED


# The namespace a case body executes in. Built explicitly rather than from
# `globals()` so that adding a private helper here does not silently widen the
# vocabulary a case may use.
NAMESPACE = {
    "table": table,
    "col": col,
    "lit": lit,
    "count_star": count_star,
    "int64": int64,
    "int32": int32,
    "float64": float64,
    "string": string,
    "bool_": bool_,
    "Int64Type": Int64Type,
    "Int32Type": Int32Type,
    "Float64Type": Float64Type,
    "StringType": StringType,
    "BoolType": BoolType,
    "Upper": Upper,
    "Lower": Lower,
    "Strip": Strip,
    "StringLength": StringLength,
    "StartsWith": StartsWith,
    "EndsWith": EndsWith,
    "Like": Like,
    "ILike": ILike,
    "IsNull": IsNull,
    "NotNull": NotNull,
    "CaseWhen": CaseWhen,
    "Coalesce": Coalesce,
    "FillNull": FillNull,
    "NumericCast": NumericCast,
    "NumToString": NumToString,
    "StringToNum": StringToNum,
    "BoolToNum": BoolToNum,
    "NumToBool": NumToBool,
    "JOIN_INNER": JOIN_INNER,
    "JOIN_LEFT": JOIN_LEFT,
    "JOIN_RIGHT": JOIN_RIGHT,
    "JOIN_FULL": JOIN_FULL,
    "JOIN_SEMI": JOIN_SEMI,
    "JOIN_ANTI": JOIN_ANTI,
    "JOIN_ALL": JOIN_ALL,
}
