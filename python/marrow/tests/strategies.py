# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Hypothesis strategies shared by the ``test_properties_*.py`` suites.

Everything here generates **PyArrow** objects: PyArrow is the reference, so the
inputs are built by it and handed to marrow over the C Data Interface, and the
answers are compared back in PyArrow.

Lengths lean on the boundaries marrow's layout cares about: buffers are 64-byte
aligned and validity is bit-packed, so 63/64/65 elements (and 127/128/129) put
the last value on either side of a word or allocation edge.

Two settings profiles are registered: ``fast`` (the default, for CI) and
``thorough``. Select one with ``MARROW_HYPOTHESIS_PROFILE=thorough``.
"""

import math
import os
from decimal import Context, Decimal

import pyarrow as pa
import pyarrow.compute as pc
from hypothesis import HealthCheck, settings
from hypothesis import strategies as st

_SUPPRESSED = [
    HealthCheck.too_slow,
    HealthCheck.data_too_large,
    HealthCheck.filter_too_much,
    # `tmp_path` is shared by a test's examples; each overwrites its file.
    HealthCheck.function_scoped_fixture,
]
settings.register_profile(
    "fast",
    max_examples=25,
    deadline=None,
    suppress_health_check=_SUPPRESSED,
    print_blob=True,
)
settings.register_profile(
    "thorough",
    max_examples=1000,
    deadline=None,
    suppress_health_check=_SUPPRESSED,
    print_blob=True,
)
settings.load_profile(os.environ.get("MARROW_HYPOTHESIS_PROFILE", "fast"))


# ── types ──────────────────────────────────────────────────────────────────

SIGNED_TYPES = [pa.int8(), pa.int16(), pa.int32(), pa.int64()]
UNSIGNED_TYPES = [pa.uint8(), pa.uint16(), pa.uint32(), pa.uint64()]
INTEGER_TYPES = SIGNED_TYPES + UNSIGNED_TYPES
FLOAT_TYPES = [pa.float16(), pa.float32(), pa.float64()]
NUMERIC_TYPES = INTEGER_TYPES + FLOAT_TYPES
STRING_TYPES = [pa.string(), pa.large_string(), pa.string_view()]
BINARY_TYPES = [pa.binary(), pa.large_binary(), pa.binary_view(), pa.binary(3)]
TEMPORAL_TYPES = [
    pa.date32(),
    pa.date64(),
    pa.time32("s"),
    pa.time32("ms"),
    pa.time64("us"),
    pa.time64("ns"),
    pa.timestamp("s"),
    pa.timestamp("ms"),
    pa.timestamp("us", "UTC"),
    pa.timestamp("ns", "America/New_York"),
    pa.duration("s"),
    pa.duration("ns"),
]
DECIMAL_TYPES = [
    pa.decimal32(7, 2),
    pa.decimal64(15, 3),
    pa.decimal128(38, 10),
    pa.decimal256(50, 5),
]
INTERVAL_TYPES = [pa.month_day_nano_interval()]

FLAT_TYPES = (
    [pa.null(), pa.bool_()]
    + NUMERIC_TYPES
    + STRING_TYPES
    + BINARY_TYPES
    + TEMPORAL_TYPES
    + DECIMAL_TYPES
    + INTERVAL_TYPES
)

flat_types = st.sampled_from(FLAT_TYPES)


def _struct_of(children):
    return st.lists(children, min_size=1, max_size=3).map(
        lambda ts: pa.struct([pa.field(f"f{i}", t) for i, t in enumerate(ts)])
    )


def nested_types(leaves=flat_types, max_leaves=3, large=True, fixed=True):
    """Leaves wrapped in list / large_list / fixed_size_list / struct / map,
    with bounded nesting depth. `large` and `fixed` switch large_list and
    fixed_size_list off for a consumer that cannot take them."""

    def wrap(inner):
        options = [
            inner.map(pa.list_),
            _struct_of(inner),
            st.tuples(st.sampled_from([pa.string(), pa.int32()]), inner).map(
                lambda p: pa.map_(*p)
            ),
        ]
        if large:
            options.append(inner.map(pa.large_list))
        if fixed:
            options.append(
                st.tuples(inner, st.integers(1, 3)).map(lambda p: pa.list_(*p))
            )
        return st.one_of(options)

    return st.recursive(leaves, wrap, max_leaves=max_leaves)


dictionary_types = st.builds(
    pa.dictionary,
    st.sampled_from(INTEGER_TYPES),
    st.sampled_from([pa.string(), pa.int64(), pa.float64(), pa.binary()]),
)


def all_types(leaves=flat_types):
    return st.one_of(leaves, nested_types(leaves), dictionary_types)


# ── values ────────────────────────────────────────────────────────────────

BOUNDARY_LENGTHS = (0, 1, 63, 64, 65, 127, 128, 129)


def lengths(max_size=200):
    """A length, half the time exactly on a 64-element boundary."""
    return st.one_of(st.sampled_from(BOUNDARY_LENGTHS), st.integers(0, max_size))


def _int_bounds(t):
    w = t.bit_width
    if pa.types.is_signed_integer(t):
        return -(2 ** (w - 1)), 2 ** (w - 1) - 1
    return 0, 2**w - 1


_WIDE = Context(prec=100)
_UNIT = {"s": 1, "ms": 10**3, "us": 10**6, "ns": 10**9}


def values(t):
    """A strategy for one non-null Python value `pa.array(..., type=t)` takes."""
    if pa.types.is_null(t):
        return st.none()
    if pa.types.is_boolean(t):
        return st.booleans()
    if pa.types.is_integer(t):
        return st.integers(*_int_bounds(t))
    if pa.types.is_floating(t):
        return st.floats(width=t.bit_width)
    if pa.types.is_fixed_size_binary(t) and not pa.types.is_decimal(t):
        return st.binary(min_size=t.byte_width, max_size=t.byte_width)
    if (
        pa.types.is_string(t)
        or pa.types.is_large_string(t)
        or pa.types.is_string_view(t)
    ):
        return st.text(max_size=12)
    if (
        pa.types.is_binary(t)
        or pa.types.is_large_binary(t)
        or pa.types.is_binary_view(t)
    ):
        return st.binary(max_size=12)
    if pa.types.is_date32(t):
        return st.integers(-100_000, 100_000)
    if pa.types.is_date64(t):
        return st.integers(-100_000, 100_000).map(lambda d: d * 86_400_000)
    if pa.types.is_time(t):
        return st.integers(0, 86_400 * _UNIT[t.unit] - 1)
    if pa.types.is_timestamp(t) or pa.types.is_duration(t):
        # Seconds stay within what milliseconds can hold: Parquet has no
        # second unit, and both writers scale to it.
        bound = 2**62 // (1000 if t.unit == "s" else 1)
        return st.integers(-bound, bound)
    if pa.types.is_decimal(t):
        bound = 10**t.precision - 1
        # A wide context: the default 28 digits would round a decimal256.
        return st.integers(-bound, bound).map(
            lambda i: Decimal(i).scaleb(-t.scale, _WIDE)
        )
    if t == pa.month_day_nano_interval():
        return st.tuples(
            st.integers(-(2**31), 2**31 - 1),
            st.integers(-(2**31), 2**31 - 1),
            st.integers(-(2**63), 2**63 - 1),
        )
    if pa.types.is_fixed_size_list(t):
        child = maybe_null(t.value_type)
        return st.lists(child, min_size=t.list_size, max_size=t.list_size)
    if pa.types.is_map(t):
        return st.lists(
            st.tuples(values(t.key_type), maybe_null(t.item_type)), max_size=3
        )
    if pa.types.is_list(t) or pa.types.is_large_list(t):
        return st.lists(maybe_null(t.value_type), max_size=4)
    if pa.types.is_struct(t):
        return st.fixed_dictionaries({f.name: maybe_null(f.type) for f in t})
    if pa.types.is_dictionary(t):
        return values(t.value_type)
    raise NotImplementedError(f"no value strategy for {t}")


def maybe_null(t):
    """A value of `t`, or None about a quarter of the time."""
    v = values(t)
    return st.one_of(st.none(), v, v, v)


@st.composite
def arrays(draw, dtype=all_types(), size=None, nullable=True):
    """A PyArrow array of a drawn type. With `nullable`, half the arrays carry
    nulls; the other half have none, and so no validity bitmap."""
    t = draw(dtype) if isinstance(dtype, st.SearchStrategy) else dtype
    n = draw(size if size is not None else lengths())
    has_nulls = nullable and draw(st.booleans())
    if pa.types.is_dictionary(t):
        # A small pool, so values repeat and the index type never overflows.
        pool = draw(st.lists(values(t.value_type), min_size=1, max_size=8))
        v = st.sampled_from(pool)
        item = st.one_of(st.none(), v, v, v) if has_nulls else v
    else:
        item = maybe_null(t) if has_nulls else values(t)
    return pa.array(draw(st.lists(item, min_size=n, max_size=n)), type=t)


@st.composite
def sliced_arrays(draw, dtype=all_types(), size=None, nullable=True):
    """An array whose offset is nonzero: generated longer, then sliced.

    Offsets of 1, 7, 8, 63, 64 and 65 put the slice start on and around a
    validity byte and a 64-byte buffer boundary."""
    n = draw(size if size is not None else lengths())
    pre = draw(st.one_of(st.sampled_from([1, 7, 8, 63, 64, 65]), st.integers(1, 70)))
    post = draw(st.integers(0, 9))
    full = draw(arrays(dtype, size=st.just(pre + n + post), nullable=nullable))
    return full.slice(pre, n)


def any_arrays(dtype=all_types(), size=None, nullable=True):
    """Either a whole array or a slice with a nonzero offset."""
    return st.one_of(
        arrays(dtype, size, nullable), sliced_arrays(dtype, size, nullable)
    )


# ── comparison ─────────────────────────────────────────────────────────────


def _canon(arr):
    """`arr` rewritten so that `Array.equals` is exact equality.

    `equals` calls NaN unequal to itself and -0.0 equal to 0.0; here floats
    become their bit patterns (with every NaN collapsed to one), and nested
    types are rebuilt around their canonicalised children."""
    t = arr.type
    if pa.types.is_floating(t):
        wide = arr.cast(pa.float64()) if pa.types.is_float16(t) else arr
        nan = pa.scalar(math.nan, wide.type)
        clean = pc.if_else(pc.is_nan(wide), nan, wide)
        return clean.view(pa.int64() if wide.type == pa.float64() else pa.int32())
    if pa.types.is_dictionary(t):
        return _canon(arr.dictionary_decode())
    mask = arr.is_null() if arr.null_count else None
    if pa.types.is_struct(t):
        children = [_canon(arr.field(i)) for i in range(t.num_fields)]
        fields = [pa.field(f.name, c.type, f.nullable) for f, c in zip(t, children)]
        return pa.StructArray.from_arrays(children, fields=fields, mask=mask)
    if pa.types.is_fixed_size_list(t):
        n = t.list_size
        child = _canon(arr.values.slice(arr.offset * n, len(arr) * n))
        return pa.FixedSizeListArray.from_arrays(child, n, mask=mask)
    if pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_map(t):
        # Rebased to a zero offset (`from_arrays` refuses a mask over sliced
        # offsets), and a map rebuilt as a list of its entry structs, since
        # `MapArray.from_arrays` takes no mask.
        offsets = arr.offsets.to_pylist()
        base = offsets[0]
        child = _canon(arr.values.slice(base, offsets[-1] - base))
        rebased = pa.array([o - base for o in offsets], arr.offsets.type)
        cls = pa.LargeListArray if pa.types.is_large_list(t) else pa.ListArray
        return cls.from_arrays(rebased, child, mask=mask)
    return arr


def assert_same(got, want, check_type=True):
    """Exact equality of two PyArrow arrays: same type, same nulls, same
    values, NaN equal to NaN and -0.0 distinct from 0.0."""
    if isinstance(got, pa.ChunkedArray):
        got = got.combine_chunks()
    if isinstance(want, pa.ChunkedArray):
        want = want.combine_chunks()
    if check_type:
        assert got.type == want.type, f"type {got.type} != {want.type}"
    assert len(got) == len(want), f"length {len(got)} != {len(want)}"
    assert got.null_count == want.null_count, (
        f"null_count {got.null_count} != {want.null_count}"
    )
    cg, cw = _canon(got), _canon(want)
    assert cg.equals(cw), f"\n got: {got}\nwant: {want}"


def has_nan(arr):
    """Whether a float array holds a NaN anywhere."""
    if not pa.types.is_floating(arr.type):
        return False
    wide = arr.cast(pa.float64()) if pa.types.is_float16(arr.type) else arr
    return pc.any(pc.is_nan(wide)).as_py() is True
