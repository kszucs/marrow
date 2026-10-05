# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Differential properties: marrow's kernels against ``pyarrow.compute``.

Every property draws PyArrow inputs (often sliced, so the offset is nonzero),
runs the same operation in both libraries and demands the same answer. The
comparison is exact — NaN equal to NaN, -0.0 distinct from 0.0 — except for
floating-point aggregates, which may sum in a different order: those compare
within ``_FLOAT_TOL`` (1e-9) times the largest input magnitude times the
input length (the squared magnitude for a variance, the result itself for a
product).

When PyArrow raises, marrow must raise too. When PyArrow does not implement an
operation for a type the case is skipped, since there is nothing to compare.

Divergences found by these properties are pinned as strict ``xfail`` cases at
the bottom of the file, each with its minimal input.
"""

import math

import pyarrow as pa
import pyarrow.compute as pc
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

import marrow as ma
import marrow.compute as mc
from marrow import col
from marrow.tests.strategies import (
    BINARY_TYPES,
    DECIMAL_TYPES,
    FLOAT_TYPES,
    INTEGER_TYPES,
    INTERVAL_TYPES,
    NUMERIC_TYPES,
    STRING_TYPES,
    TEMPORAL_TYPES,
    any_arrays,
    assert_same,
    dictionary_types,
    flat_types,
    has_nan,
    lengths,
    nested_types,
)

# ── helpers ────────────────────────────────────────────────────────────────


def to_pa(marrow_array):
    return pa.array(marrow_array)


def reference(fn, *args, **kwargs):
    """PyArrow's answer, or the exception it raised. `None` when PyArrow does
    not implement the operation for these types."""
    try:
        return fn(*args, **kwargs)
    except (pa.ArrowNotImplementedError, NotImplementedError):
        return None
    except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowIndexError) as exc:
        return exc


@st.composite
def pairs(draw, dtypes):
    """Two arrays of one type and one length, either possibly sliced."""
    t = draw(st.sampled_from(dtypes))
    n = draw(lengths())
    a = draw(any_arrays(st.just(t), size=st.just(n)))
    b = draw(any_arrays(st.just(t), size=st.just(n)))
    return a, b


def batch(**columns):
    """A marrow RecordBatch over PyArrow columns."""
    return ma.record_batch(pa.record_batch(columns))


def has_int_min_over_minus_one(a, b):
    """Whether a signed division hits INT_MIN / -1 anywhere."""
    if not pa.types.is_signed_integer(a.type):
        return False
    low = -(2 ** (a.type.bit_width - 1))
    hits = pc.and_(pc.equal(a, low), pc.equal(b, -1))
    return pc.any(hits).as_py() is True


# ── arithmetic ─────────────────────────────────────────────────────────────


@given(pairs(NUMERIC_TYPES), st.sampled_from(["add", "subtract", "multiply"]))
def test_arithmetic_matches_pyarrow(pair, op):
    """add / subtract / multiply: the unchecked kernels, which wrap on integer
    overflow in both libraries."""
    a, b = pair
    want = reference(getattr(pc, op), a, b)
    assume(want is not None and not isinstance(want, Exception))
    got = to_pa(getattr(mc, op)(ma.array(a), ma.array(b)))
    assert_same(got, want)


@given(pairs(NUMERIC_TYPES))
def test_divide_matches_pyarrow(pair):
    """divide: integer division truncates toward zero; a zero integer divisor
    raises in both. INT_MIN / -1 is excluded here and pinned below."""
    a, b = pair
    assume(not has_int_min_over_minus_one(a, b))
    want = reference(pc.divide, a, b)
    assume(want is not None)
    if isinstance(want, Exception):
        with pytest.raises(ma.ArrowException):
            mc.divide(ma.array(a), ma.array(b))
    else:
        assert_same(to_pa(mc.divide(ma.array(a), ma.array(b))), want)


# ── comparison ─────────────────────────────────────────────────────────────

_COMPARISONS = ["equal", "not_equal", "less", "less_equal", "greater", "greater_equal"]


@given(
    pairs(NUMERIC_TYPES + TEMPORAL_TYPES + DECIMAL_TYPES + INTERVAL_TYPES),
    st.sampled_from(_COMPARISONS),
)
def test_comparison_matches_pyarrow(pair, op):
    """The six comparisons over numeric, temporal, decimal and interval types:
    a NaN compares unequal to everything, itself included."""
    a, b = pair
    want = reference(getattr(pc, op), a, b)
    assume(want is not None and not isinstance(want, Exception))
    got = to_pa(getattr(mc, op)(ma.array(a), ma.array(b)))
    assert_same(got, want)


_EXPR_COMPARISONS = {
    "equal": lambda x, y: x == y,
    "not_equal": lambda x, y: x != y,
    "less": lambda x, y: x < y,
    "less_equal": lambda x, y: x <= y,
    "greater": lambda x, y: x > y,
    "greater_equal": lambda x, y: x >= y,
}


@given(pairs(STRING_TYPES[:2] + BINARY_TYPES[:2]), st.sampled_from(_COMPARISONS))
def test_string_comparison_matches_pyarrow(pair, op):
    """String and binary comparison, bytewise. `marrow.compute` refuses
    non-primitive types, so this goes through the expression layer."""
    a, b = pair
    got = to_pa(_EXPR_COMPARISONS[op](col("a"), col("b")).execute(batch(a=a, b=b)))
    assert_same(got, getattr(pc, op)(a, b))


# ── boolean ────────────────────────────────────────────────────────────────

_BOOLEAN = {
    "and": (lambda x, y: x & y, pc.and_kleene),
    "or": (lambda x, y: x | y, pc.or_kleene),
    "xor": (lambda x, y: x ^ y, pc.xor),
}


@given(pairs([pa.bool_()]), st.sampled_from(sorted(_BOOLEAN)))
def test_boolean_matches_pyarrow(pair, op):
    """`&` and `|` are Kleene (``false & null`` is false), `^` propagates
    nulls."""
    a, b = pair
    build, reference_fn = _BOOLEAN[op]
    got = to_pa(build(col("a"), col("b")).execute(batch(a=a, b=b)))
    assert_same(got, reference_fn(a, b))


@given(any_arrays(st.just(pa.bool_())))
def test_invert_matches_pyarrow(a):
    got = to_pa((~col("a")).execute(batch(a=a)))
    assert_same(got, pc.invert(a))


@given(any_arrays(st.just(pa.bool_())), st.sampled_from(["any", "all"]))
def test_any_all_match_pyarrow(a, op):
    """any / all skipping nulls. Compared with ``min_count=0``: marrow answers
    a value for an empty or all-null input where PyArrow's default answers
    null — pinned below."""
    got = getattr(mc, op)(ma.array(a))
    assert got == getattr(pc, op)(a, min_count=0).as_py()


@given(any_arrays(), st.sampled_from(["is_null", "is_valid", "drop_null"]))
def test_null_kernels_match_pyarrow(a, op):
    """is_null / is_valid / drop_null over every type, nested included."""
    want = reference(getattr(pc, op), a)
    assume(want is not None)
    got = to_pa(getattr(mc, op)(ma.array(a)))
    assert_same(got, want)


# ── aggregates ─────────────────────────────────────────────────────────────

_FLOAT_TOL = 1e-9

# Finite and bounded so that summation order cannot overflow, plus the three
# specials whose propagation does not depend on the order.
_bounded_floats = st.one_of(
    st.floats(-1e6, 1e6),
    st.sampled_from([math.nan, math.inf, -math.inf]),
)


@st.composite
def aggregate_inputs(draw, dtypes):
    t = draw(st.sampled_from(dtypes))
    n = draw(lengths(150))
    if pa.types.is_floating(t) and t != pa.float16():
        item = st.one_of(st.none(), _bounded_floats, _bounded_floats)
        pre = draw(st.integers(0, 70))
        raw = draw(st.lists(item, min_size=pre + n, max_size=pre + n))
        return pa.array(raw, t).slice(pre, n)
    return draw(any_arrays(st.just(t), size=st.just(n)))


def _close(got, want, values):
    """`got` equals `want`, within `_FLOAT_TOL` of the input's magnitude when
    they are floats."""
    if want is None or got is None:
        return got is None and want is None
    if isinstance(want, float) or isinstance(got, float):
        if math.isnan(want):
            return math.isnan(got)
        if math.isinf(want):
            return got == want
        scale = max(
            [1.0] + [abs(v) for v in values if v is not None and math.isfinite(v)]
        )
        return abs(got - want) <= _FLOAT_TOL * scale * max(1, len(values))
    return got == want


def aggregate(arr, verb):
    out = ma.memtable(batch(x=arr)).aggregate(r=(verb, "x")).collect()
    return out.to_pydict()["r"][0]


_EXACT_AGGREGATES = {
    "sum": pc.sum,
    "min": pc.min,
    "max": pc.max,
    "count": pc.count,
    "count_distinct": pc.count_distinct,
    "product": pc.product,
}


@given(
    aggregate_inputs(INTEGER_TYPES + FLOAT_TYPES[1:]),
    st.sampled_from(sorted(_EXACT_AGGREGATES)),
)
def test_aggregate_matches_pyarrow(arr, verb):
    """sum / product / min / max / count / count_distinct, through the lazy
    aggregate. Integer results are exact (both wrap on overflow); float sums
    and products may differ by summation order, within `_FLOAT_TOL`."""
    if verb == "count_distinct" and pa.types.is_floating(arr.type):
        # -0.0 and 0.0 count as one value in marrow and two in PyArrow. Not a
        # bug: hashing canonicalises -0.0 on purpose, so that it groups with
        # 0.0 as SQL requires (kernels/hashing.mojo).
        arr = pc.if_else(pc.equal(arr, 0), pa.scalar(0.0, arr.type), arr)
    want = reference(_EXACT_AGGREGATES[verb], arr)
    assume(want is not None and not isinstance(want, Exception))
    values = arr.to_pylist()
    if verb in ("min", "max") and pa.types.is_floating(arr.type):
        valid = [v for v in values if v is not None]
        ordered = [v for v in valid if not math.isnan(v)]
        # The identity is the largest finite float rather than an infinity, so
        # all-NaN or all-(+/-)inf input answers it: pinned below.
        identity = math.inf if verb == "min" else -math.inf
        assume(not valid or (ordered and not all(v == identity for v in ordered)))
    got = aggregate(arr, verb)
    if verb in ("sum", "product") and pa.types.is_unsigned_integer(arr.type):
        # marrow accumulates unsigned input as int64 where PyArrow uses uint64
        # (pinned below); the bits agree, so compare them.
        got = None if got is None else got % 2**64
    if verb == "product" and pa.types.is_floating(arr.type):
        # A product's rounding is relative to the product, not the inputs.
        w = want.as_py()
        assert _close(got, w, [w])
    else:
        assert _close(got, want.as_py(), values), (got, want.as_py())


_STATISTICS = {
    "mean": pc.mean,
    "variance": lambda a: pc.variance(a, ddof=0),
    "var_samp": lambda a: pc.variance(a, ddof=1),
    "stddev": lambda a: pc.stddev(a, ddof=0),
    "stddev_samp": lambda a: pc.stddev(a, ddof=1),
}


@given(
    aggregate_inputs(
        [pa.int8(), pa.int16(), pa.int32(), pa.uint8(), pa.uint16(), pa.float64()]
    ),
    st.sampled_from(sorted(_STATISTICS)),
)
def test_statistics_match_pyarrow(arr, verb):
    """mean / variance / stddev, population and sample. Variance compares
    within `_FLOAT_TOL` of the squared magnitude."""
    want = _STATISTICS[verb](arr).as_py()
    got = aggregate(arr, verb)
    values = arr.to_pylist()
    if verb in ("variance", "var_samp") and want is not None and math.isfinite(want):
        scale = max(
            [1.0] + [abs(v) for v in values if v is not None and math.isfinite(v)]
        )
        values = [scale * scale]
    assert _close(got, want, values), (got, want)


@st.composite
def grouped(draw):
    """A key column with a small domain (so groups collide), nulls included,
    and an int64 value column."""
    kt = draw(st.sampled_from([pa.int8(), pa.int64(), pa.string()]))
    domain = [1, 2, 3, -4] if pa.types.is_integer(kt) else ["a", "b", "", "é"]
    n = draw(lengths(150))
    keys = draw(
        st.lists(st.one_of(st.none(), st.sampled_from(domain)), min_size=n, max_size=n)
    )
    vals = draw(
        st.lists(st.one_of(st.none(), st.integers(-1000, 1000)), min_size=n, max_size=n)
    )
    return pa.array(keys, kt), pa.array(vals, pa.int64())


@given(grouped())
def test_group_by_matches_pyarrow(kv):
    """Grouped sum / count / min / max, nulls forming their own group. Groups
    are compared as a mapping, since neither library promises an order."""
    k, v = kv
    want = (
        pa.table({"k": k, "v": v})
        .group_by("k")
        .aggregate([("v", "sum"), ("v", "count"), ("v", "min"), ("v", "max")])
    )
    got = (
        ma.memtable(batch(k=k, v=v))
        .aggregate(
            by=["k"],
            v_sum=("sum", "v"),
            v_count=("count", "v"),
            v_min=("min", "v"),
            v_max=("max", "v"),
        )
        .collect()
        .to_pydict()
    )
    want = want.to_pydict()
    names = ["v_sum", "v_count", "v_min", "v_max"]

    def as_map(d):
        return {key: tuple(d[n][i] for n in names) for i, key in enumerate(d["k"])}

    assert as_map(got) == as_map(want)
    assert len(got["k"]) == len(want["k"])


# ── join ───────────────────────────────────────────────────────────────────

_JOIN_KINDS = [
    "inner",
    "left outer",
    "right outer",
    "full outer",
    "left semi",
    "left anti",
]


@st.composite
def join_inputs(draw):
    kt = draw(st.sampled_from([pa.int32(), pa.int64(), pa.string()]))
    domain = [1, 2, 3] if pa.types.is_integer(kt) else ["a", "b", "c"]
    key = st.one_of(st.none(), st.sampled_from(domain))
    ln = draw(st.integers(0, 70))
    rn = draw(st.integers(0, 70))
    lk = pa.array(draw(st.lists(key, min_size=ln, max_size=ln)), kt)
    rk = pa.array(draw(st.lists(key, min_size=rn, max_size=rn)), kt)
    return lk, rk


@given(join_inputs(), st.sampled_from(_JOIN_KINDS))
def test_join_matches_pyarrow(keys, kind):
    """Hash join on one key with duplicates and nulls on both sides; a null key
    matches nothing. Rows compare as a multiset of (left row id, right row id)
    pairs, which is what a join decides."""
    lk, rk = keys
    left = pa.record_batch({"k": lk, "lid": pa.array(range(len(lk)), pa.int64())})
    right = pa.record_batch({"k": rk, "rid": pa.array(range(len(rk)), pa.int64())})
    want = (
        pa.Table.from_batches([left])
        .join(pa.Table.from_batches([right]), keys="k", join_type=kind)
        .to_pydict()
    )
    got = ma.record_batch(left).join(ma.record_batch(right), "k", join_type=kind)
    got = got.to_pydict()

    def rows(d):
        n = len(d["k"])
        return sorted(zip(d.get("lid", [None] * n), d.get("rid", [None] * n)), key=repr)

    assert rows(got) == rows(want)


# ── membership ─────────────────────────────────────────────────────────────


@st.composite
def membership_inputs(draw):
    t = draw(st.sampled_from([pa.int64(), pa.string()]))
    domain = [0, 1, -1, 7] if t == pa.int64() else ["", "a", "B", "é"]
    item = st.one_of(st.none(), st.sampled_from(domain))
    n = draw(lengths(150))
    pre = draw(st.integers(0, 70))
    arr = pa.array(draw(st.lists(item, min_size=pre + n, max_size=pre + n)), t)
    value_set = draw(st.lists(item, max_size=4))
    return arr.slice(pre, n), value_set


@given(membership_inputs())
def test_is_in_matches_pyarrow(inputs):
    """is_in with PyArrow's default ``skip_nulls=False``: a null matches a null
    in the value set."""
    arr, value_set = inputs
    value_set = pa.array(value_set, arr.type)
    want = pc.is_in(arr, value_set=value_set)
    got = to_pa(col("x").isin(ma.array(value_set)).execute(batch(x=arr)))
    assert_same(got, want)


# ── cast ───────────────────────────────────────────────────────────────────

_CAST_TYPES = (
    NUMERIC_TYPES
    + [pa.bool_(), pa.string(), pa.large_string()]
    + [
        pa.date32(),
        pa.date64(),
        pa.timestamp("s"),
        pa.timestamp("ms"),
        pa.timestamp("us"),
    ]
    + [pa.time32("ms"), pa.time64("ns"), pa.duration("ms")]
    + [pa.decimal128(20, 4)]
)


@st.composite
def cast_inputs(draw):
    src = draw(st.sampled_from(_CAST_TYPES))
    dst = draw(st.sampled_from(_CAST_TYPES))
    if pa.types.is_string(src) or pa.types.is_large_string(src):
        # Strings that parse as often as not.
        text = st.one_of(
            st.integers(-300, 300).map(str),
            st.floats(allow_nan=True).map(str),
            st.sampled_from(["true", "false", "1970-01-02", "", " 1", "1e3", "x"]),
        )
        n = draw(lengths(100))
        arr = pa.array(
            draw(st.lists(st.one_of(st.none(), text), min_size=n, max_size=n)), src
        )
    else:
        arr = draw(any_arrays(st.just(src), size=lengths(100)))
    return arr, dst, draw(st.booleans())


def _known_cast_divergence(src, dst, safe):
    """The pairs left out of the property: pinned at the bottom of the file
    unless the comment says why they are not a bug."""
    integers = pa.types.is_integer(src) and pa.types.is_integer(dst)
    sign_change = integers and (
        # signed -> unsigned of equal or greater width, unsigned -> signed of
        # equal width: the pairs where the range check is missing.
        (
            pa.types.is_signed_integer(src)
            and pa.types.is_unsigned_integer(dst)
            and dst.bit_width >= src.bit_width
        )
        or (
            pa.types.is_unsigned_integer(src)
            and pa.types.is_signed_integer(dst)
            and dst.bit_width == src.bit_width
        )
    )
    # PyArrow does not check int -> float16 precision (2049 becomes 2048.0)
    # where it does for float32 and float64; marrow checks all three.
    to_half = safe and pa.types.is_integer(src) and pa.types.is_float16(dst)
    timestamp_to_date = pa.types.is_timestamp(src) and (
        pa.types.is_date64(dst) or (safe and pa.types.is_date32(dst))
    )
    string_to_number = pa.types.is_string(src) or pa.types.is_large_string(src)
    string_to_number = string_to_number and (pa.types.is_integer(dst) or not safe)
    truncating_downscale = (
        not safe
        and pa.types.is_timestamp(src)
        and pa.types.is_timestamp(dst)
        and _UNITS.index(dst.unit) < _UNITS.index(src.unit)
    )
    timestamp_to_time = pa.types.is_timestamp(src) and pa.types.is_time(dst)
    # Float and temporal formatting is a representation choice ("0.0"
    # against "0", or a timestamp with or without its fractional digits).
    float_to_string = (pa.types.is_floating(src) or pa.types.is_temporal(src)) and (
        pa.types.is_string(dst) or pa.types.is_large_string(dst)
    )
    return (
        (safe and sign_change)
        or to_half
        or float_to_string
        or timestamp_to_date
        or timestamp_to_time
        or string_to_number
        or truncating_downscale
    )


_UNITS = ["s", "ms", "us", "ns"]


@given(cast_inputs())
def test_cast_matches_pyarrow(inputs):
    """Cast across numeric, bool, string, temporal and decimal types, safe and
    unsafe. Where PyArrow raises, marrow must raise; where PyArrow succeeds,
    marrow must agree or refuse the pair as unsupported.

    Not compared: a string that fails to parse under ``safe=False``, which
    PyArrow raises on and marrow answers null for — a deliberate difference."""
    arr, dst, safe = inputs
    if _known_cast_divergence(arr.type, dst, safe):
        return  # pinned below
    if pa.types.is_floating(arr.type) and pa.types.is_integer(dst):
        # Unsafe: out of range (or NaN) is undefined behaviour in PyArrow's
        # float -> int, so only in-range values compare. Safe: marrow misses
        # some out-of-range values (pinned below), so those are left out too.
        low, high = -(2.0 ** (dst.bit_width - 1)), 2.0 ** (dst.bit_width - 1)
        if pa.types.is_unsigned_integer(dst):
            low, high = 0.0, 2.0**dst.bit_width
        if not all(v is None or low <= v < high for v in arr.to_pylist()):
            return
    if pa.types.is_floating(arr.type) and pa.types.is_decimal(dst):
        # Past 2**53 the scaled value is rounded: pinned below.
        limit = 2.0**53 / 10**dst.scale
        if not all(v is None or not abs(v) >= limit for v in arr.to_pylist()):
            return
    want = reference(pc.cast, arr, dst, safe=safe)
    assume(want is not None)
    m = ma.array(arr)
    if isinstance(want, Exception) and "Precision is not great enough" in str(want):
        # PyArrow refuses int -> decimal by the types alone; marrow checks the
        # values, which is the more permissive and not a wrong answer.
        return
    if (
        isinstance(want, Exception)
        and pa.types.is_integer(arr.type)
        and pa.types.is_floating(dst)
    ):
        # PyArrow refuses any integer past 2**53 (2**24 for float32); marrow
        # refuses only the ones the float cannot hold exactly, so a success
        # must be exact.
        try:
            got = to_pa(mc.cast(m, dst, safe=safe))
        except ma.ArrowInvalid:
            return
        assert got.to_pylist() == arr.to_pylist()
        return
    if isinstance(want, Exception):
        with pytest.raises(ma.ArrowException):
            to_pa(mc.cast(m, dst, safe=safe))
        return
    try:
        got = to_pa(mc.cast(m, dst, safe=safe))
    except (ma.ArrowNotImplementedError, ma.ArrowTypeError):
        return  # an unsupported pair is a gap, not a wrong answer
    if pa.types.is_decimal(arr.type) and pa.types.is_floating(dst):
        # PyArrow scales in the target precision and can land an ulp away
        # from the nearest float; marrow is allowed to be the closer one.
        tol = 1e-6 if dst.bit_width == 32 else 1e-15
        pairs_ = zip(got.to_pylist(), want.to_pylist())
        assert all(
            g is None if w is None else math.isclose(g, w, rel_tol=tol)
            for g, w in pairs_
        )
        return
    assert_same(got, want)


# ── selection ──────────────────────────────────────────────────────────────


@st.composite
def with_mask(draw):
    arr = draw(any_arrays())
    mask = draw(any_arrays(st.just(pa.bool_()), size=st.just(len(arr))))
    return arr, mask


@given(with_mask())
def test_filter_matches_pyarrow(inputs):
    """filter over every type; a null in the mask drops the row."""
    arr, mask = inputs
    want = reference(pc.filter, arr, mask)
    assume(want is not None)
    got = to_pa(mc.filter(ma.array(arr), ma.array(mask)))
    assert_same(got, want)


@st.composite
def with_indices(draw):
    arr = draw(any_arrays())
    n = len(arr)
    it = draw(st.sampled_from([pa.int32(), pa.int64(), pa.uint8(), pa.uint32()]))
    if it == pa.uint8():
        assume(n <= 256)
    index = st.none() if n == 0 else st.one_of(st.none(), st.integers(0, n - 1))
    size = draw(lengths(150))
    indices = pa.array(draw(st.lists(index, min_size=size, max_size=size)), it)
    return arr, indices


@given(with_indices())
def test_take_matches_pyarrow(inputs):
    """take over every type with in-bounds indices; a null index gives a
    null."""
    arr, indices = inputs
    want = reference(pc.take, arr, indices)
    assume(want is not None)
    got = to_pa(mc.take(ma.array(arr), ma.array(indices)))
    assert_same(got, want)


@given(
    st.one_of(flat_types, nested_types(), dictionary_types),
    st.integers(1, 4),
    st.data(),
)
def test_concat_matches_pyarrow(t, count, data):
    """concat of one to four arrays of one type, sliced or not."""
    arrays = [data.draw(any_arrays(st.just(t), size=lengths(70))) for _ in range(count)]
    want = reference(pa.concat_arrays, arrays)
    assume(want is not None and not isinstance(want, Exception))
    got = to_pa(ma.concat_arrays([ma.array(a) for a in arrays]))
    assert_same(got, want)


def test_concat_dictionary_keeps_dictionary():
    d = pa.array(["x", None, "y"]).dictionary_encode()
    got = to_pa(ma.concat_arrays([ma.array(d), ma.array(d)]))
    assert got.to_pylist() == ["x", None, "y", "x", None, "y"]


# ── sort ───────────────────────────────────────────────────────────────────

_SORTABLE = (
    NUMERIC_TYPES
    + [pa.bool_(), pa.string(), pa.large_string(), pa.binary()]
    + TEMPORAL_TYPES
    + DECIMAL_TYPES
)


def _unsigned_zero(arr):
    """-0.0 rewritten as 0.0: the two sort as equal, so with an unstable sort
    either may come first."""
    if not pa.types.is_floating(arr.type):
        return arr
    return pc.if_else(pc.equal(arr, 0), pa.scalar(0.0, arr.type), arr)


@given(
    any_arrays(st.sampled_from(_SORTABLE)),
    st.sampled_from(["ascending", "descending"]),
    st.sampled_from(["at_start", "at_end"]),
)
def test_sort_indices_matches_pyarrow(arr, order, null_placement):
    """sort_indices orders the values and puts nulls where asked. Compared by
    the values the indices select: marrow's sort is not stable (pinned below),
    so equal values may come back in another order. NaN placement is pinned
    separately too."""
    assume(not has_nan(arr))
    want = reference(
        pc.sort_indices, arr, sort_keys=[("", order)], null_placement=null_placement
    )
    assume(want is not None)
    got = to_pa(
        mc.sort_indices(ma.array(arr), [("", order)], null_placement=null_placement)
    )
    assert sorted(got.to_pylist()) == list(range(len(arr)))
    assert_same(_unsigned_zero(arr.take(got)), _unsigned_zero(arr.take(want)))


@given(
    any_arrays(st.sampled_from(_SORTABLE)),
    st.sampled_from(["ascending", "descending"]),
    st.sampled_from(["at_start", "at_end"]),
)
def test_sort_matches_pyarrow(arr, order, null_placement):
    """sort returns the values sort_indices would select."""
    assume(not has_nan(arr))
    indices = reference(
        pc.sort_indices, arr, sort_keys=[("", order)], null_placement=null_placement
    )
    assume(indices is not None)
    got = to_pa(mc.sort(ma.array(arr), [("", order)], null_placement=null_placement))
    assert_same(_unsigned_zero(got), _unsigned_zero(arr.take(indices)))


@st.composite
def sort_by_inputs(draw):
    n = draw(lengths(150))
    a = draw(any_arrays(st.sampled_from([pa.int8(), pa.string()]), size=st.just(n)))
    b = draw(any_arrays(st.sampled_from([pa.int64(), pa.float64()]), size=st.just(n)))
    assume(not has_nan(b))
    # -0.0 sorts before 0.0 rather than equal to it: pinned below.
    b = _unsigned_zero(b)
    keys = draw(
        st.lists(
            st.tuples(
                st.sampled_from(["a", "b"]),
                st.sampled_from(["ascending", "descending"]),
            ),
            min_size=1,
            max_size=2,
            unique_by=lambda k: k[0],
        )
    )
    return pa.record_batch({"a": a, "b": b, "i": pa.array(range(n), pa.int64())}), keys


@given(sort_by_inputs(), st.sampled_from(["at_start", "at_end"]))
def test_sort_by_matches_pyarrow(inputs, null_placement):
    """RecordBatch.sort_by over one or two keys, lexicographic. Compared on
    the key columns only, since the sort is not stable."""
    rb, keys = inputs
    want = rb.sort_by(keys, null_placement=null_placement)
    got = pa.record_batch(
        ma.record_batch(rb).sort_by(keys, null_placement=null_placement)
    )
    for name, _ in keys:
        assert_same(_unsigned_zero(got.column(name)), _unsigned_zero(want.column(name)))


# ── strings ────────────────────────────────────────────────────────────────

# Letters with interesting case mappings, multi-byte code points and the
# whitespace that trimming has to recognise.
# 'ß' is left out: its case mapping is pinned below.
_ALPHABET = "aAzZ0 \t\néÉΣσ€😀"
_texts = st.text(alphabet=_ALPHABET, max_size=8)


@st.composite
def string_arrays(draw):
    t = draw(st.sampled_from([pa.string(), pa.large_string()]))
    n = draw(lengths(130))
    pre = draw(st.integers(0, 70))
    raw = draw(
        st.lists(st.one_of(st.none(), _texts), min_size=pre + n, max_size=pre + n)
    )
    return pa.array(raw, t).slice(pre, n)


_UNARY_STRINGS = {
    "upper": pc.utf8_upper,
    "lower": pc.utf8_lower,
    "length": pc.binary_length,
    "char_length": pc.utf8_length,
    "reverse": pc.utf8_reverse,
    "strip": pc.utf8_trim_whitespace,
    "lstrip": pc.utf8_ltrim_whitespace,
    "rstrip": pc.utf8_rtrim_whitespace,
    "capitalize": pc.utf8_capitalize,
}


@given(string_arrays(), st.sampled_from(sorted(_UNARY_STRINGS)))
def test_unary_string_matches_pyarrow(arr, verb):
    """The unary string verbs against their ``utf8_*`` counterparts."""
    got = to_pa(getattr(col("s"), verb)().execute(batch(s=arr)))
    want = _UNARY_STRINGS[verb](arr)
    assert got.to_pylist() == want.to_pylist()


_STRING_PREDICATES = {
    "startswith": pc.starts_with,
    "endswith": pc.ends_with,
    "contains": pc.match_substring,
}


@given(string_arrays(), _texts, st.sampled_from(sorted(_STRING_PREDICATES)))
def test_string_predicate_matches_pyarrow(arr, pattern, verb):
    """startswith / endswith / contains with a literal pattern."""
    got = to_pa(getattr(col("s"), verb)(pattern).execute(batch(s=arr)))
    assert_same(got, _STRING_PREDICATES[verb](arr, pattern=pattern))


# ── pinned divergences ─────────────────────────────────────────────────────


@pytest.mark.xfail(
    strict=True,
    reason="divide: INT_MIN / -1 wraps to INT_MIN; pyarrow.compute.divide answers 0",
)
def test_divide_int_min_by_minus_one():
    a = pa.array([-128], pa.int8())
    b = pa.array([-1], pa.int8())
    assert (
        to_pa(mc.divide(ma.array(a), ma.array(b))).to_pylist()
        == pc.divide(a, b).to_pylist()
    )


@pytest.mark.xfail(
    strict=True,
    reason="any/all: an empty or all-null input answers False/True; "
    "pyarrow's default min_count=1 answers null",
)
@pytest.mark.parametrize("op", ["any", "all"])
@pytest.mark.parametrize("values", [[], [None]])
def test_any_all_empty_is_null(op, values):
    a = pa.array(values, pa.bool_())
    assert getattr(mc, op)(ma.array(a)) == getattr(pc, op)(a).as_py()


@pytest.mark.xfail(
    strict=True,
    reason="take: an out-of-bounds or negative index yields null instead of raising",
)
@pytest.mark.parametrize("index", [1, -1])
def test_take_out_of_bounds_raises(index):
    arr = ma.array(pa.array([10], pa.int64()))
    with pytest.raises(ma.ArrowException):
        mc.take(arr, ma.array(pa.array([index], pa.int64())))


@pytest.mark.xfail(
    strict=True,
    reason="sort_indices orders NaN as the largest value; PyArrow keeps NaN next "
    "to the nulls whatever the order (just before them at_end, just after them "
    "at_start)",
)
@pytest.mark.parametrize(
    "order, null_placement",
    [("descending", "at_end"), ("ascending", "at_start")],
)
def test_sort_indices_nan_placement(order, null_placement):
    arr = pa.array([math.nan, 1.0, None])
    want = pc.sort_indices(arr, sort_keys=[("", order)], null_placement=null_placement)
    got = mc.sort_indices(ma.array(arr), [("", order)], null_placement=null_placement)
    assert to_pa(got).to_pylist() == want.to_pylist()


@pytest.mark.xfail(
    strict=True,
    reason="RecordBatch.sort_by: nulls default to first; pyarrow defaults to at_end",
)
def test_sort_by_default_null_placement():
    rb = pa.record_batch({"a": pa.array([None, 1], pa.int64())})
    assert ma.record_batch(rb).sort_by("a").to_pydict() == rb.sort_by("a").to_pydict()


@pytest.mark.xfail(
    strict=True,
    reason="sort_indices is not stable from 32 elements on: equal values come "
    "back out of input order",
)
@pytest.mark.parametrize("n", [32, 63, 64, 65])
def test_sort_indices_is_stable(n):
    arr = pa.array([0] * n, pa.int8())
    assert to_pa(mc.sort_indices(ma.array(arr))).to_pylist() == list(range(n))


@pytest.mark.xfail(
    strict=True,
    reason="cast(safe=True) from signed to an unsigned integer at least as wide, "
    "or from unsigned to the signed integer of the same width, wraps instead of "
    "raising",
)
@pytest.mark.parametrize(
    "value, src, dst",
    [
        (-1, pa.int8(), pa.uint8()),
        (-1, pa.int8(), pa.uint16()),
        (-1, pa.int32(), pa.uint64()),
        (-5, pa.int32(), pa.uint32()),
        (-1, pa.int64(), pa.uint64()),
        (200, pa.uint8(), pa.int8()),
        (2**63, pa.uint64(), pa.int64()),
    ],
)
def test_cast_sign_change_is_checked(value, src, dst):
    with pytest.raises(ma.ArrowInvalid):
        mc.cast(ma.array(pa.array([value], src)), dst, safe=True)


@pytest.mark.xfail(
    strict=True,
    reason="upper/capitalize apply the full case mapping ('ß' -> 'SS', and "
    "capitalize gives 'SS', not even the titlecase 'Ss'); PyArrow's utf8_upper "
    "and utf8_capitalize map one code point to one ('ß' -> 'ẞ')",
)
@pytest.mark.parametrize("verb", ["upper", "capitalize"])
def test_case_mapping_sharp_s(verb):
    arr = pa.array(["ß"])
    got = to_pa(getattr(col("s"), verb)().execute(batch(s=arr)))
    assert got.to_pylist() == _UNARY_STRINGS[verb](arr).to_pylist()


@pytest.mark.xfail(
    strict=True,
    reason="cast timestamp -> date64 rescales to milliseconds without flooring "
    "to midnight, so the date64 keeps its time of day",
)
@pytest.mark.parametrize("safe", [False, True])
def test_cast_timestamp_to_date64_floors_to_midnight(safe):
    arr = pa.array([1], pa.timestamp("s"))
    got = to_pa(mc.cast(ma.array(arr), pa.date64(), safe=safe))
    assert got.view(pa.int64()).to_pylist() == [0]


@pytest.mark.xfail(
    strict=True,
    reason="cast(safe=True) timestamp -> date32 raises when the time of day is "
    "nonzero; PyArrow drops the time of day",
)
def test_cast_timestamp_to_date32_safe():
    arr = pa.array([1], pa.timestamp("s"))
    got = to_pa(mc.cast(ma.array(arr), pa.date32(), safe=True))
    assert got.to_pylist() == pc.cast(arr, pa.date32()).to_pylist()


@pytest.mark.xfail(
    strict=True,
    reason="cast(safe=True) string -> integer wraps a parsed value that does not "
    "fit the target ('128' -> int8 gives -128)",
)
@pytest.mark.parametrize(
    "text, dst", [("128", pa.int8()), ("-1", pa.uint8()), ("300", pa.uint8())]
)
def test_cast_string_to_integer_out_of_range(text, dst):
    with pytest.raises(ma.ArrowInvalid):
        mc.cast(ma.array(pa.array([text])), dst, safe=True)


@pytest.mark.xfail(
    strict=True,
    reason="cast(safe=False) to a coarser timestamp unit floors a negative tick "
    "(-1 ms -> -1 s); PyArrow truncates toward zero (0 s)",
)
def test_cast_timestamp_downscale_truncates():
    arr = pa.array([-1], pa.timestamp("ms"))
    got = to_pa(mc.cast(ma.array(arr), pa.timestamp("s"), safe=False))
    assert got.view(pa.int64()).to_pylist() == [0]


@pytest.mark.xfail(
    strict=True,
    reason="cast timestamp -> time rescales the whole tick instead of taking the "
    "time of day: -1 s gives -1000 ms and 2,091,084 s gives 2,091,084,000 ms, "
    "both outside a day (and safe=True raises an overflow for the second)",
)
@pytest.mark.parametrize("tick", [-1, 2_091_084])
def test_cast_timestamp_to_time_of_day(tick):
    arr = pa.array([tick], pa.timestamp("s"))
    got = to_pa(mc.cast(ma.array(arr), pa.time32("ms"), safe=False))
    want = pc.cast(arr, pa.time32("ms"), safe=False)
    # Compared as storage: `to_pylist` folds an out-of-range time into a day.
    assert got.view(pa.int32()).to_pylist() == want.view(pa.int32()).to_pylist()


@pytest.mark.xfail(
    strict=True,
    reason="cast float -> decimal scales in floating point, so a double that is "
    "an exact integer comes back with a rounding error once value * 10**scale "
    "passes 2**53",
)
def test_cast_float_to_decimal_is_exact():
    arr = pa.array([14411518807587.0])
    got = to_pa(mc.cast(ma.array(arr), pa.decimal128(20, 4), safe=False))
    assert got.to_pylist() == pc.cast(arr, pa.decimal128(20, 4), safe=False).to_pylist()


@pytest.mark.xfail(
    strict=True,
    reason="sum/product of unsigned integers accumulate as int64; PyArrow answers "
    "uint64, so a total past 2**63 comes back negative",
)
@pytest.mark.parametrize("verb", ["sum", "product"])
def test_unsigned_sum_is_uint64(verb):
    arr = pa.array([2**63, 1], pa.uint64())
    assert aggregate(arr, verb) == getattr(pc, verb)(arr).as_py()


@pytest.mark.xfail(
    strict=True,
    reason="cast(safe=True) float -> int misses out-of-range values: float64 -> "
    "8/16-bit integers and float16 -> any integer wrap, and float64 2**63 -> int64 "
    "and float16 inf -> int32 saturate",
)
@pytest.mark.parametrize(
    "value, src, dst",
    [
        (128.0, pa.float64(), pa.int8()),
        (40000.0, pa.float64(), pa.int16()),
        (256.0, pa.float64(), pa.uint8()),
        (2.0**63, pa.float64(), pa.int64()),
        (128.0, pa.float16(), pa.int8()),
        (math.inf, pa.float16(), pa.int32()),
    ],
)
def test_cast_float_to_int_out_of_range(value, src, dst):
    arr = pa.array([value]).cast(src)
    with pytest.raises(ma.ArrowInvalid):
        mc.cast(ma.array(arr), dst, safe=True)


@pytest.mark.xfail(
    strict=True,
    reason="min/max start from the largest finite float instead of an infinity: "
    "over only NaN they answer +/-FLT_MAX (or DBL_MAX) instead of NaN, and "
    "min([inf]) answers FLT_MAX",
)
@pytest.mark.parametrize(
    "verb, values",
    [
        ("min", [math.nan, None]),
        ("max", [math.nan, None]),
        ("min", [math.inf]),
        ("max", [-math.inf]),
    ],
)
@pytest.mark.parametrize("dtype", [pa.float32(), pa.float64()])
def test_min_max_identity(verb, values, dtype):
    arr = pa.array(values, dtype)
    got = aggregate(arr, verb)
    want = getattr(pc, verb)(arr).as_py()
    assert got == want or (math.isnan(got) and math.isnan(want))


@pytest.mark.xfail(
    strict=True,
    reason="sort_by orders -0.0 before 0.0 instead of treating them as equal, so "
    "a later key no longer decides between them",
)
def test_sort_by_signed_zero_ties():
    rb = pa.record_batch({"b": pa.array([0.0, -0.0]), "a": pa.array([0, 1])})
    keys = [("b", "ascending"), ("a", "ascending")]
    got = pa.record_batch(ma.record_batch(rb).sort_by(keys, null_placement="at_end"))
    assert got.column("a").to_pylist() == [0, 1]
