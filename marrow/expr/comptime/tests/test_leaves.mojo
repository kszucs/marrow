"""Every family's leaves — a column, a literal and a parameter.

Each case runs a plan, so a leaf is checked where it is used: a column must
come back with its dtype, a literal must broadcast to the batch, and a
parameter must read its value from the execution and refuse a binding of the
wrong dtype.
"""

from std.testing import assert_equal, assert_raises, assert_true

from ....arrays import DictionaryArray, ListArray, StructArray
from ....builders import (
    BinaryLikeBuilder,
    Decimal128Builder,
    FixedSizeBinaryBuilder,
    FixedSizeListBuilder,
    Int64Builder,
    ListBuilder,
    StructBuilder,
    YearMonthIntervalBuilder,
    array,
    nulls,
)
from ....dtypes import (
    BinaryType,
    Decimal128Type,
    LargeStringType,
    DynType,
    MonthDayNanoIntervalType,
    binary,
    bool_,
    date32,
    decimal128,
    dictionary,
    field,
    fixed_size_binary_,
    fixed_size_list_,
    int32,
    int64,
    large_binary,
    large_list_,
    large_string,
    list_,
    millisecond,
    null,
    second,
    string,
    struct_,
    timestamp,
    year_month_interval,
)
from ....scalars import (
    BinaryScalar,
    BoolScalar,
    Date32Scalar,
    Decimal128Scalar,
    DynScalar,
    Int32Scalar,
    Int64Scalar,
    LargeStringScalar,
    ListScalar,
    MonthDayNanoIntervalScalar,
    NullScalar,
    StringScalar,
    StructScalar,
    TimestampScalar,
    YearMonthIntervalScalar,
)
from ....tabular import record_batch
from ...builders import array_length, col, lit, param, table
from ...bindings import ParamSpec
from ...logical import DynRelation, References, Value


def _table() raises -> DynRelation:
    """Three rows, so a broadcast has a length to reach."""
    return table(record_batch([array([1, 2, 3], int64).copy()], names=["a"]))


# ---------------------------------------------------------------------------
# bool
# ---------------------------------------------------------------------------


def test_bool_literal_broadcasts() raises:
    var got = _table().project(["t"], [lit(True, bool_)]).execute()
    assert_true(got.columns[0].as_bool() == array([True, True, True]))


def test_bool_param_reads_its_binding_and_default() raises:
    var plan = _table().project(["on"], [param("on", bool_, default=False)])
    assert_true(
        plan.execute().columns[0].as_bool() == array([False, False, False])
    )
    var bound = plan.execute(bindings={"on": BoolScalar(True).to_dyn()})
    assert_true(bound.columns[0].as_bool() == array([True, True, True]))


def test_bool_param_gates_a_filter() raises:
    """A parameter is an operand like any other: a Kleene `AND` over a fused
    comparison and a boolean parameter."""
    var plan = _table().filter(
        (col("a", int64) > lit(1, int64)) & param("on", bool_)
    )
    var on = plan.execute(bindings={"on": BoolScalar(True).to_dyn()})
    assert_true(on.columns[0].as_int64() == array([2, 3], int64))
    var off = plan.execute(bindings={"on": BoolScalar(False).to_dyn()})
    assert_equal(off.num_rows(), 0)


# ---------------------------------------------------------------------------
# string
# ---------------------------------------------------------------------------


def test_string_param_reads_its_binding_and_default() raises:
    var plan = _table().project(
        ["s"], [param("s", string, default=String("x"))]
    )
    assert_true(plan.execute().columns[0].as_string() == array(["x", "x", "x"]))
    var bound = plan.execute(bindings={"s": StringScalar("y").to_dyn()})
    assert_true(bound.columns[0].as_string() == array(["y", "y", "y"]))


def test_string_param_compares_against_a_column() raises:
    var t = table(
        record_batch([array(["a", "b", "a"]).copy()], names=["s"])
    ).filter(col("s", string) == param("want", string))
    var got = t.execute(bindings={"want": StringScalar("a").to_dyn()})
    assert_equal(got.num_rows(), 2)


def test_large_string_leaves_keep_their_dtype() raises:
    var plan = _table().project(
        ["lit", "param"],
        [lit(String("x"), large_string), param("s", large_string)],
    )
    var got = plan.execute(bindings={"s": LargeStringScalar("y").to_dyn()})
    for i in range(2):
        assert_true(got.columns[i].dtype() == DynType(large_string))
    assert_equal(got.columns[0].as_large_string()[2].value(), "x")
    assert_equal(got.columns[1].as_large_string()[0].value(), "y")
    with assert_raises(contains="'s' is large_string"):
        _ = plan.execute(bindings={"s": StringScalar("y").to_dyn()})


# ---------------------------------------------------------------------------
# temporal
# ---------------------------------------------------------------------------


def test_temporal_literal_broadcasts_with_its_dtype() raises:
    var got = _table().project(["d"], [lit(19000, date32())]).execute()
    assert_true(got.columns[0].dtype() == DynType(date32()))
    ref d = got.columns[0].as_date32()
    assert_equal(len(d), 3)
    assert_true(d[2].value() == 19000)


def test_temporal_param_reads_its_binding() raises:
    var ts = timestamp(millisecond)
    var plan = _table().project(["t"], [param("t", ts)])
    var got = plan.execute(
        bindings={"t": TimestampScalar(Optional(Int64(5000)), ts).to_dyn()}
    )
    assert_true(got.columns[0].dtype() == DynType(ts))
    assert_true(got.columns[0].as_timestamp()[1].value() == 5000)


# ---------------------------------------------------------------------------
# decimal
# ---------------------------------------------------------------------------


def _decimals() raises -> DynRelation:
    var d = Decimal128Builder(decimal128(10, 2), 3)
    d.append(Scalar[Decimal128Type.native](150))
    d.append_null()
    d.append(Scalar[Decimal128Type.native](-25))
    return table(record_batch([d.finish().to_dyn()], names=["d"]))


def test_decimal_column_keeps_precision_and_scale() raises:
    var got = (
        _decimals().project(["d"], [col("d", decimal128(10, 2))]).execute()
    )
    assert_true(got.columns[0].dtype() == DynType(decimal128(10, 2)))
    ref d = got.columns[0].as_decimal128()
    assert_true(d[0].value() == 150)
    assert_true(not d.is_valid(1))


def test_decimal_column_reaches_a_fused_null_test() raises:
    var got = (
        _decimals()
        .project(["n"], [col("d", decimal128(10, 2)).is_null()])
        .execute()
    )
    assert_true(got.columns[0].as_bool() == array([False, True, False]))


def test_decimal_literal_broadcasts_from_int_and_scalar() raises:
    var dt = decimal128(38, 4)
    var wide = Scalar[Decimal128Type.native](Int64.MAX) * 10
    var got = (
        _table()
        .project(
            ["small", "wide"],
            [
                lit(12345, dt),
                lit(Decimal128Scalar(Optional(wide), dt.copy())),
            ],
        )
        .execute()
    )
    assert_true(got.columns[0].dtype() == DynType(dt))
    assert_true(got.columns[0].as_decimal128()[2].value() == 12345)
    assert_true(got.columns[1].as_decimal128()[0].value() == wide)


def test_decimal_param_reads_its_default() raises:
    var plan = _table().project(
        ["p"], [param("p", decimal128(10, 2), default=Optional(Int128(7)))]
    )
    assert_true(plan.execute().columns[0].as_decimal128()[0].value() == 7)


# ---------------------------------------------------------------------------
# interval
# ---------------------------------------------------------------------------


def test_interval_column_literal_and_param() raises:
    var months = YearMonthIntervalBuilder(year_month_interval(), 2)
    months.append(Int32(3))
    months.append(Int32(14))
    var plan = table(
        record_batch([months.finish().to_dyn()], names=["m"])
    ).project(
        ["col", "lit", "param"],
        [
            col("m", year_month_interval()),
            lit(12, year_month_interval()),
            param("p", year_month_interval()),
        ],
    )
    var got = plan.execute(
        bindings={"p": YearMonthIntervalScalar(Int32(6)).to_dyn()}
    )
    assert_true(got.columns[0].as_year_month_interval()[1].value() == 14)
    assert_true(got.columns[1].as_year_month_interval()[0].value() == 12)
    assert_true(got.columns[2].as_year_month_interval()[1].value() == 6)
    assert_true(got.columns[1].dtype() == DynType(year_month_interval()))


def test_month_day_nano_literal_takes_the_full_width() raises:
    var packed = Scalar[MonthDayNanoIntervalType.native](Int64.MAX) * 4
    var got = (
        _table()
        .project(["i"], [lit(MonthDayNanoIntervalScalar(packed))])
        .execute()
    )
    assert_true(
        got.columns[0].as_month_day_nano_interval()[0].value() == packed
    )


# ---------------------------------------------------------------------------
# every parameter checks its binding
# ---------------------------------------------------------------------------


def test_a_param_refuses_a_binding_of_another_dtype() raises:
    """Another scalar kind, and the same kind with another unit, scale or
    struct — each would otherwise reach an unchecked downcast."""
    var ints = _table().filter(col("a", int64) > param("min", int64))
    with assert_raises(contains="'min' is int64"):
        _ = ints.execute(bindings={"min": Int32Scalar(1).to_dyn()})

    var ts = _table().project(["t"], [param("t", timestamp(millisecond))])
    var seconds = TimestampScalar(Optional(Int64(5)), timestamp(second))
    with assert_raises(contains="'t' is timestamp[ms]"):
        _ = ts.execute(bindings={"t": seconds^.to_dyn()})

    var dec = _table().project(["p"], [param("p", decimal128(10, 2))])
    var scale3 = Decimal128Scalar(
        Optional(Scalar[Decimal128Type.native](7)), decimal128(10, 3)
    )
    with assert_raises(contains="'p' is decimal128[10, 2]"):
        _ = dec.execute(bindings={"p": scale3^.to_dyn()})

    var st = _structs()
    var structs = _table().project(
        ["s"], [param("s", struct_([field("a", int64)]))]
    )
    with assert_raises(contains="'s' is struct"):
        _ = structs.execute(bindings={"s": DynScalar(st[0])})


def test_a_null_binding_is_refused() raises:
    """A null of the right kind, as opposed to a scalar of the wrong one."""
    var x = _table().project(["x"], [param("x", int64)])
    with assert_raises(contains="'x' was bound to null"):
        _ = x.execute(
            bindings={"x": Int64Scalar(Optional[Int64](None)).to_dyn()}
        )
    var y = _table().project(["y"], [param("y", string)])
    with assert_raises(contains="'y' was bound to null"):
        _ = y.execute(bindings={"y": StringScalar.null().to_dyn()})
    var z = _table().project(["z"], [param("z", bool_)])
    with assert_raises(contains="'z' was bound to null"):
        _ = z.execute(bindings={"z": BoolScalar.null().to_dyn()})


# ---------------------------------------------------------------------------
# binary
# ---------------------------------------------------------------------------


def _bytes(text: String) -> List[UInt8]:
    return List[UInt8](text.as_bytes())


def test_binary_column_literal_and_param() raises:
    var b = BinaryLikeBuilder[BinaryType](2)
    b.append("ab")
    b.append("cd")
    var plan = table(record_batch([b.finish().to_dyn()], names=["b"])).project(
        ["col", "lit", "param"],
        [
            col("b", binary),
            lit(_bytes("hi"), binary),
            param("p", binary, default=Optional(_bytes("zz"))),
        ],
    )
    var got = plan.execute(bindings={"p": BinaryScalar("xy").to_dyn()})
    for i in range(3):
        assert_true(got.columns[i].dtype() == DynType(binary))
    assert_equal(got.columns[0].as_binary()[1].value(), "cd")
    assert_equal(got.columns[1].as_binary()[1].value(), "hi")
    assert_equal(got.columns[2].as_binary()[0].value(), "xy")
    var unbound = plan.execute()
    assert_equal(unbound.columns[2].as_binary()[1].value(), "zz")


def test_large_binary_literal_keeps_its_offset_width() raises:
    var got = (
        _table().project(["b"], [lit(_bytes("hi"), large_binary)]).execute()
    )
    assert_true(got.columns[0].dtype() == DynType(large_binary))
    assert_equal(got.columns[0].as_large_binary()[2].value(), "hi")


# ---------------------------------------------------------------------------
# fixed-size binary
# ---------------------------------------------------------------------------


def test_fixed_size_binary_column_literal_and_param() raises:
    var fb = FixedSizeBinaryBuilder(2)
    fb.append(Span(_bytes("ab")))
    fb.append(Span(_bytes("cd")))
    var arr = fb.finish()
    var plan = table(record_batch([arr.copy().to_dyn()], names=["f"])).project(
        ["col", "lit", "param"],
        [
            col("f", fixed_size_binary_(2)),
            lit(arr[0]),
            param("p", fixed_size_binary_(2)),
        ],
    )
    var got = plan.execute(bindings={"p": DynScalar(arr[1])})
    for i in range(3):
        assert_true(got.columns[i].dtype() == DynType(fixed_size_binary_(2)))
    assert_true(
        got.columns[0].as_fixed_size_binary()[1].value() == _bytes("cd")
    )
    assert_true(
        got.columns[1].as_fixed_size_binary()[1].value() == _bytes("ab")
    )
    assert_true(
        got.columns[2].as_fixed_size_binary()[0].value() == _bytes("cd")
    )


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


def _lists() raises -> ListArray:
    """[[1, 2], [3]]."""
    var lists = ListBuilder(Int64Builder())
    var child_any = lists.values()
    ref child = child_any.as_int64()
    child.append(1)
    child.append(2)
    lists.append_valid()
    child.append(3)
    lists.append_valid()
    return lists.finish()


def test_list_column_literal_and_param() raises:
    var xs = _lists()
    var plan = table(record_batch([xs.copy().to_dyn()], names=["xs"])).project(
        ["col", "lit", "param"],
        [
            col("xs", list_(int64)),
            lit(xs[0], list_(int64)),
            param("p", list_(int64)),
        ],
    )
    var got = plan.execute(bindings={"p": DynScalar(xs[1])})
    for i in range(3):
        assert_true(got.columns[i].dtype() == DynType(list_(int64)))
    assert_true(got.columns[0].as_list() == xs)
    ref lit_col = got.columns[1].as_list()
    assert_equal(len(lit_col), 2)
    assert_true(lit_col.values().as_int64() == array([1, 2, 1, 2], int64))
    assert_true(
        got.columns[2].as_list().values().as_int64() == array([3, 3], int64)
    )


def test_list_literal_is_consumed_by_a_list_node() raises:
    """`array_length` binds its operand, so a literal list reaches it as a
    broadcast column."""
    var xs = _lists()
    var got = (
        table(record_batch([xs.copy().to_dyn()], names=["xs"]))
        .project(["n"], [array_length(lit(xs[0], list_(int64)))])
        .execute()
    )
    assert_true(got.columns[0].as_int32() == array([2, 2], int32))


def test_list_literal_refuses_another_list_dtype() raises:
    var xs = _lists()
    with assert_raises(contains="large_list"):
        _ = lit(xs[0], large_list_(int64))


def test_null_list_literal_broadcasts_nulls() raises:
    var nothing = ListScalar(
        dtype=list_(int64).to_dyn(),
        value=array(int64).to_dyn(),
        is_valid=False,
    )
    var got = _table().project(["xs"], [lit(nothing^, list_(int64))]).execute()
    assert_equal(got.columns[0].null_count(), 3)
    assert_equal(len(got.columns[0].as_list().values()), 0)


# ---------------------------------------------------------------------------
# fixed-size list
# ---------------------------------------------------------------------------


def test_fixed_size_list_column_literal_and_param() raises:
    var ints = Int64Builder()
    var fsl = FixedSizeListBuilder(ints^, list_size=2)
    var child_any = fsl.values()
    ref child = child_any.as_int64()
    child.append(1)
    child.append(2)
    fsl.append_valid()
    child.append(3)
    child.append(4)
    fsl.append_valid()
    var arr = fsl.finish()
    var dt = fixed_size_list_(int64, 2)
    var plan = table(record_batch([arr.copy().to_dyn()], names=["v"])).project(
        ["col", "lit", "param"],
        [col("v", dt.copy()), lit(arr[1], dt.copy()), param("p", dt.copy())],
    )
    var got = plan.execute(bindings={"p": DynScalar(arr[0])})
    for i in range(3):
        assert_true(got.columns[i].dtype() == DynType(dt.copy()))
    assert_true(
        got.columns[1].as_fixed_size_list().values().as_int64()
        == array([3, 4, 3, 4], int64)
    )
    assert_true(
        got.columns[2].as_fixed_size_list().values().as_int64()
        == array([1, 2, 1, 2], int64)
    )


# ---------------------------------------------------------------------------
# struct
# ---------------------------------------------------------------------------


def _structs() raises -> StructArray:
    """{a: 1, b: 10}, {a: 2, b: 20}."""
    var sb = StructBuilder([field("a", int32), field("b", int32)], capacity=2)
    sb.field_builder(0).as_int32().append(1)
    sb.field_builder(0).as_int32().append(2)
    sb.field_builder(1).as_int32().append(10)
    sb.field_builder(1).as_int32().append(20)
    sb.append_valid()
    sb.append_valid()
    return sb.finish()


def test_struct_column_literal_and_param() raises:
    var st = _structs()
    var dt = struct_([field("a", int32), field("b", int32)])
    var plan = table(record_batch([st.copy().to_dyn()], names=["s"])).project(
        ["col", "lit", "param"],
        [col("s", dt.copy()), lit(st[1]), param("p", dt.copy())],
    )
    var got = plan.execute(bindings={"p": DynScalar(st[0])})
    for i in range(3):
        assert_true(got.columns[i].dtype() == DynType(dt.copy()))
    assert_true(
        got.columns[1].as_struct().field(1).as_int32() == array([20, 20], int32)
    )
    assert_true(
        got.columns[2].as_struct().field(0).as_int32() == array([1, 1], int32)
    )


def test_null_struct_literal_is_null_in_every_row() raises:
    var dt = struct_([field("a", int32)])
    var got = (
        _table()
        .project(["s"], [lit(StructScalar.null(dt^.to_dyn())).is_null()])
        .execute()
    )
    assert_true(got.columns[0].as_bool() == array([True, True, True]))


# ---------------------------------------------------------------------------
# dictionary
# ---------------------------------------------------------------------------


def test_dictionary_column_literal_and_param() raises:
    var dt = dictionary(int32, string)
    var d = DictionaryArray(
        dtype=dt.copy().to_dyn(),
        length=3,
        nulls=0,
        offset=0,
        indices=array([0, 2, 1], int32).to_dyn(),
        values=array(["a", "b", "c"]).to_dyn(),
    )
    var plan = table(record_batch([d.copy().to_dyn()], names=["d"])).project(
        ["col", "lit", "param"],
        [col("d", dt.copy()), lit(d[1]), param("p", dt.copy())],
    )
    var got = plan.execute(bindings={"p": DynScalar(d[2])})
    for i in range(3):
        assert_true(got.columns[i].dtype() == DynType(dt.copy()))
    assert_equal(
        got.columns[0].as_dictionary()[1].value().as_string().value(), "c"
    )
    assert_equal(
        got.columns[1].as_dictionary()[2].value().as_string().value(), "c"
    )
    assert_equal(
        got.columns[2].as_dictionary()[0].value().as_string().value(), "b"
    )


# ---------------------------------------------------------------------------
# null
# ---------------------------------------------------------------------------


def test_null_column_literal_and_param() raises:
    var plan = table(
        record_batch([nulls(3, null.to_dyn())], names=["n"])
    ).project(
        ["col", "lit", "param", "is_null"],
        [
            col("n", null),
            lit(NullScalar()),
            param("p", null),
            col("n", null).is_null(),
        ],
    )
    var got = plan.execute(bindings={"p": NullScalar().to_dyn()})
    for i in range(3):
        assert_true(got.columns[i].dtype() == DynType(null))
        assert_equal(len(got.columns[i]), 3)
    assert_true(got.columns[3].as_bool() == array([True, True, True]))


# ---------------------------------------------------------------------------
# the erased ladder
# ---------------------------------------------------------------------------


def test_an_erased_temporal_literal_broadcasts() raises:
    var scalar = Date32Scalar(Optional(Int32(19000)), date32())
    var got = _table().project(["d"], [lit(DynScalar(scalar^))]).execute()
    assert_true(got.columns[0].as_date32()[2].value() == 19000)


def test_an_erased_struct_literal_broadcasts() raises:
    """A struct reached through `DynScalar.to_array` — which recurses into it
    for each field."""
    var st = _structs()
    var got = _table().project(["s"], [lit(DynScalar(st[0]))]).execute()
    assert_true(
        got.columns[0].as_struct().field(1).as_int32()
        == array([10, 10, 10], int32)
    )


# ---------------------------------------------------------------------------
# broadcasts that must keep the declared dtype
# ---------------------------------------------------------------------------


def test_struct_literal_keeps_a_large_string_field() raises:
    """The field value is read from a `large_string` column, so its scalar
    reports `large_string` and the field broadcasts at that width."""
    var strings = BinaryLikeBuilder[LargeStringType](1)
    strings.append("x")
    var dt = struct_([field("s", large_string)])
    var value = List[DynScalar]()
    value.append(strings.finish()[0].to_dyn())
    var scalar = StructScalar(dtype=dt^.to_dyn(), value=value^, is_valid=True)
    var got = _table().project(["st"], [lit(scalar^)]).execute()
    ref child = got.columns[0].as_struct().field(0)
    assert_true(child.dtype() == DynType(large_string))
    assert_equal(child.as_large_string()[1].value(), "x")


def test_a_fixed_size_list_scalar_of_the_wrong_length_raises() raises:
    """A null fixed-size list still owns its slots; a scalar that holds none
    would build a child shorter than the dtype says."""
    var dt = fixed_size_list_(int64, 2)
    var short = ListScalar(
        dtype=dt.copy().to_dyn(), value=array(int64).to_dyn(), is_valid=False
    )
    var plan = _table().project(["v"], [lit(short^, dt^)])
    with assert_raises(contains="2 elements"):
        _ = plan.execute()


def test_lit_refuses_a_null_decimal_scalar() raises:
    """It used to read the null's placeholder and answer a valid zero."""
    var null_decimal = Decimal128Scalar(
        Optional[Scalar[Decimal128Type.native]](None), decimal128(10, 2)
    )
    with assert_raises(contains="null"):
        _ = lit(null_decimal^)


# ---------------------------------------------------------------------------
# every family's parameter declares itself
# ---------------------------------------------------------------------------


def _spec[V: Value](value: V) -> ParamSpec:
    var refs = References()
    value.references(refs)
    return refs.params[0].copy()


def test_every_param_family_declares_its_spec() raises:
    """One case builds every family's parameter, so every family's
    `references` override is compiled, and pins what the command line needs
    from each: its name, dtype, shown default, and whether a token parses."""
    var n = _spec(param("n", int64, default=Int64(1), help="a number"))
    assert_equal(n.name, String("n"))
    assert_true(n.dtype == DynType(int64))
    assert_equal(n.help, String("a number"))
    assert_equal(n.default.value(), String("1"))
    assert_true(True if n.parse else False)

    var b = _spec(param("b", bool_, default=False))
    assert_true(b.dtype == DynType(bool_))
    assert_equal(b.default.value(), String("false"))
    assert_true(True if b.parse else False)

    var s = _spec(param("s", string, default=String("x")))
    assert_equal(s.default.value(), String("x"))
    assert_true(True if s.parse else False)
    assert_true(True if _spec(param("ls", large_string)).parse else False)

    var ts = _spec(param("ts", timestamp(millisecond)))
    assert_true(ts.dtype == DynType(timestamp(millisecond)))
    assert_true(not ts.default)
    assert_true(not ts.parse)

    assert_true(not _spec(param("d", decimal128(10, 2))).parse)
    assert_true(not _spec(param("i", year_month_interval())).parse)
    assert_true(not _spec(param("x", binary)).parse)
    assert_true(not _spec(param("fb", fixed_size_binary_(4))).parse)
    assert_true(not _spec(param("l", list_(int64))).parse)
    assert_true(not _spec(param("fl", fixed_size_list_(int64, 2))).parse)
    assert_true(not _spec(param("st", struct_([field("a", int64)]))).parse)
    assert_true(not _spec(param("dict", dictionary(int32, string))).parse)
    assert_true(not _spec(param("nil", null)).parse)
