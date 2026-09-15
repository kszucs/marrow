"""`references` — the one walk over what an expression reads.

A composite node's walk is derived from its fields by reflection, so these
cases build one tree per composite family and pin the exact answer: which
columns, in first-seen order, and which parameters, in walk order. A leaf that
forgot its override would fail the build rather than reach these cases.
"""

from std.testing import assert_equal, assert_raises, assert_true

from ...builders import array
from ...dtypes import (
    bool_,
    date32,
    float64,
    int64,
    list_,
    second,
    string,
    timestamp,
    uint8,
)
from ...kernels.join import JOIN_INNER
from ...tabular import RecordBatch, record_batch
from ..builders import (
    array_contains,
    array_length,
    col,
    if_else,
    is_in,
    lit,
    param,
    table,
)
from ..logical import DynRelation, DynValue, References, Value
from ..`comptime`.boolean import Not


def _refs[V: Value](value: V) -> References:
    var refs = References()
    value.references(refs)
    return refs^


def _names(params_of: References) -> List[String]:
    var out = List[String]()
    for ref p in params_of.params:
        out.append(p.name.copy())
    return out^


def _plan_names(plan: DynRelation) raises -> List[String]:
    var specs = plan.params()
    var out = List[String]()
    for ref p in specs:
        out.append(p.name.copy())
    return out^


def _batch() raises -> RecordBatch:
    return record_batch(
        [array([1, 5, 9], int64).copy(), array([2, 4, 6], int64).copy()],
        names=["a", "b"],
    )


def test_references_walk_a_numeric_tree_three_deep() raises:
    var v = Not((col("a", int64) + param("t", int64)) > lit(3, int64))
    var refs = _refs(v)
    assert_true(refs.columns == ["a"])
    assert_true(_names(refs) == ["t"])
    assert_true(v.columns() == ["a"])


def test_references_dedup_columns_in_first_seen_order() raises:
    var v = if_else(
        col("b", int64) > col("a", int64),
        col("c", int64) + param("t", int64),
        col("b", int64),
    )
    var refs = _refs(v)
    assert_true(refs.columns == ["b", "a", "c"])
    assert_true(_names(refs) == ["t"])


def test_references_walk_boolean_nodes() raises:
    var v = is_in(col("a", int64), array([1, 2], int64).to_dyn()) & col(
        "f", bool_
    )
    assert_true(_refs(v).columns == ["a", "f"])


def test_references_walk_a_cast() raises:
    var v = col("i", int64).cast(float64, safe=False) > param("x", float64)
    var refs = _refs(v)
    assert_true(refs.columns == ["i"])
    assert_true(_names(refs) == ["x"])


def test_references_skip_unused_string_slots() raises:
    var v = col("s", string).substr(col("i", int64), param("n", int64))
    var refs = _refs(v)
    assert_true(refs.columns == ["s", "i"])
    assert_true(_names(refs) == ["n"])
    assert_true(_refs(col("s", string).substr(2, 3)).columns == ["s"])
    assert_true(
        _refs(col("name", string).like(lit("p%", string))).columns == ["name"]
    )


def test_references_walk_temporal_nodes() raises:
    assert_true(_refs(col("ts", timestamp(second)).year()).columns == ["ts"])
    assert_true(
        _refs(
            col("ts", timestamp(second))
            >= col("us", timestamp(second)).date_trunc("day")
        ).columns
        == ["ts", "us"]
    )


def test_references_walk_nested_nodes() raises:
    assert_true(
        _refs(
            array_contains(col("xs", list_(int64)), col("needle", int64))
        ).columns
        == ["xs", "needle"]
    )
    assert_true(_refs(array_length(col("xs", list_(int64)))).columns == ["xs"])


def test_references_walk_aggregates_in_both_lanes() raises:
    assert_true(
        _refs(
            (col("qty", int64) * col("price", int64)).sum().alias("r")
        ).columns
        == ["qty", "price"]
    )
    assert_true(
        _refs((col("qty") * col("price")).sum().alias("r")).columns
        == ["qty", "price"]
    )
    assert_true(_refs(col("a") + col("b") + col("a")).columns == ["a", "b"])


def test_references_cross_the_box() raises:
    var boxed: DynValue = col("a", int64) > param("t", int64)
    assert_true(boxed.columns() == ["a"])
    var refs = References()
    boxed.references(refs)
    assert_true(_names(refs) == ["t"])


def test_a_numeric_param_parses_its_own_token() raises:
    var spec = _refs(param("t", int64)).params[0].copy()
    assert_true(True if spec.parse else False)
    assert_equal(spec.parse.value()("42").as_int64().value(), Int64(42))

    var narrow = _refs(param("u", uint8)).params[0].copy()
    assert_equal(narrow.parse.value()("200").as_uint8().value(), UInt8(200))
    with assert_raises(contains="out of range"):
        _ = narrow.parse.value()("300")
    with assert_raises(contains="out of range"):
        _ = narrow.parse.value()("-1")

    var text = _refs(param("s", string)).params[0].copy()
    assert_equal(
        text.parse.value()("hello").as_string().value(), String("hello")
    )
    assert_true(not _refs(param("d", date32())).params[0].parse)


def test_plan_params_walk_every_relation() raises:
    var projected: List[DynValue] = [
        col("a", int64),
        col("b", int64) + param("p_project", int64),
    ]
    var plan = (
        table(_batch())
        .filter(col("a", int64) > param("p_filter", int64))
        .project(["a", "b"], projected^)
        .aggregate(
            [(col("b", int64) * param("p_agg", int64)).sum().alias("s")],
            [col("a", int64)],
        )
        .sort_by([col("s", int64) + param("p_sort", int64)], [True])
        .limit(10)
    )
    assert_true(
        _plan_names(plan) == ["p_filter", "p_project", "p_agg", "p_sort"]
    )

    var windowed = table(_batch()).with_columns(
        ["lagged"],
        [
            col("a", int64)
            .lag()
            .over(
                partition_by=[col("b", int64) + param("p_part", int64)],
                order_by=[col("a", int64)],
            )
        ],
    )
    assert_true(_plan_names(windowed) == ["p_part"])

    var joined = (
        table(_batch())
        .filter(col("a", int64) > param("p_left", int64))
        .join(
            table(_batch()).filter(col("b", int64) > param("p_right", int64)),
            [0],
            [0],
            JOIN_INNER,
        )
    )
    assert_true(_plan_names(joined) == ["p_left", "p_right"])
    assert_true(len(table(_batch()).params()) == 0)
