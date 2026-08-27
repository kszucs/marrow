"""The string family — the one that cannot vectorise.

`NumericValue.lane[W]` and `BoolValue.lane[W]` answer `W` elements because
their storage is fixed-width. UTF-8 is not, so `StringValue.lane` answers one
`String` and every node here loops. These cases assert that the family still
composes exactly like the others: a string predicate bit-packs, feeds `And`,
and drives a `Filter` unchanged.
"""

from std.testing import assert_equal, assert_true

from ...builders import col, lit, table
from ...params import Bindings
from ....builders import array, Int64Builder
from ....arrays import StringArray
from ....dtypes import (
    DynType,
    Int64Type,
    StringType,
    int32,
    int64,
    string,
)
from ....tabular import RecordBatch, record_batch
from ...logical import DynValue
from ...logical import DynRelation, Filter, InMemoryTable, Project
from ..boolean import And
from ..leaves import Column, Literal, StringColumn, StringLiteral
from ..numeric import Gt
from ..strings import (
    EndsWith,
    ILike,
    Like,
    Lower,
    StartsWith,
    StrEq,
    StrGt,
    StrLt,
    StrNe,
    StringLength,
    Strip,
    Upper,
)


def _names() raises -> StringArray:
    var values: List[Optional[String]] = ["pear", "quince", None, "apple"]
    return array(values)


def _batch() raises -> RecordBatch:
    """A null in `name` so validity has something to report."""
    return record_batch(
        [_names().to_dyn(), array([1, 2, 3, 4], int64).copy()],
        names=["name", "a"],
    )


def test_a_string_column_evaluates_to_itself() raises:
    """A leaf hands back its own array rather than copying every byte through
    a builder — the reason the trait default is overridable."""
    var b = _batch()
    var got = (
        col("name", string)
        .evaluate(b.to_struct_array(), Bindings())
        .to_array(4)
    )
    assert_true(got.as_string() == _names())


def test_a_string_literal_stays_scalar() raises:
    var l = lit("pear", string)
    assert_true(l.evaluate(_batch().to_struct_array(), Bindings()).is_scalar())
    assert_equal(len(l.columns()), 0)


def test_string_equality_bit_packs_like_a_numeric_comparison() raises:
    """The property that lets a string predicate feed `And` and `Filter`
    unchanged: the output is a packed bool column, whatever the input width."""
    var b = _batch()
    var pred = col("name", string) == lit("pear", string)
    var got = (
        pred.evaluate(b.to_struct_array(), Bindings())
        .to_array(4)
        .as_bool()
        .copy()
    )
    assert_true(got[0].value())
    assert_true(not got[1].value())
    assert_true(got.is_null(2))  # NULL = 'pear' is NULL, not false
    assert_true(not got[3].value())


def test_a_null_string_compares_to_null_not_false() raises:
    """Null-in, null-out. The data bit is computed regardless — the loop
    compares whatever bytes are there — so validity is the only record that the
    answer is meaningless."""
    var b = _batch()
    for pred in [(col("name", string) != lit("pear", string))]:
        var got = (
            pred.evaluate(b.to_struct_array(), Bindings())
            .to_array(4)
            .as_bool()
            .copy()
        )
        assert_true(not got[0].value())  # "pear" != "pear"
        assert_true(got[1].value())  # "quince" != "pear"
        assert_true(got.is_null(2))
        assert_true(got[3].value())  # "apple" != "pear"


def test_string_ordering() raises:
    var b = _batch()
    var lt = (
        (col("name", string) < lit("pear", string))
        .evaluate(b.to_struct_array(), Bindings())
        .to_array(4)
        .as_bool()
        .copy()
    )
    assert_true(lt[3].value())  # "apple" < "pear"
    assert_true(not lt[1].value())  # "quince" > "pear"

    var gt = (
        (col("name", string) > lit("pear", string))
        .evaluate(b.to_struct_array(), Bindings())
        .to_array(4)
        .as_bool()
        .copy()
    )
    assert_true(gt[1].value())


def test_a_string_predicate_fuses_with_a_numeric_one() raises:
    """Two families in one fused subtree, which is the whole claim: `lane`
    returns different things, but both bit-pack, so `And` composes them without
    knowing either."""
    var plan = table(_batch()).filter(
        (
            (col("name", string) > lit("b", string))
            & (col("a", int64) > lit(1, int64))
        )
    )
    var out = plan.execute()
    # name > "b" and a > 1  ->  only "quince" (a=2); the null does not select
    assert_equal(out.num_rows(), 1)
    assert_true(out.columns[1].as_int64() == array([2], int64))


def test_a_string_column_projects() raises:
    var b = _batch()
    var plan = table(b.copy()).project(["who"], [col("name", string)])
    assert_true(plan.schema().fields[0].dtype == DynType(string))
    assert_true(plan.execute().columns[0].as_string() == _names())


def _others() raises -> StringArray:
    """A second nullable string column whose null sits at a *different* index
    than `_names()`, so a column-vs-column comparison has to intersect two
    distinct validity bitmaps rather than copy one."""
    var values: List[Optional[String]] = [
        "pear",  # == name[0]
        None,  # null here, name[1] valid
        "fig",  # valid here, name[2] null
        "banana",  # != name[3]
    ]
    return array(values)


def test_string_comparison_intersects_both_operand_validities() raises:
    """Every other case here is column-vs-literal, where the literal is always
    valid and the result validity is just the column's. With two nullable
    columns the null positions are disjoint, so the output is null at *both*
    — the only shape that distinguishes `Bitmap.intersect` from a copy."""
    var b = record_batch(
        [_names().to_dyn(), _others().to_dyn()], names=["name", "other"]
    )
    var got = (
        (col("name", string) == col("other", string))
        .evaluate(b.to_struct_array(), Bindings())
        .to_array(4)
        .as_bool()
        .copy()
    )
    assert_true(got[0].value())  # "pear" == "pear"
    assert_true(got.is_null(1))  # other is null
    assert_true(got.is_null(2))  # name is null
    assert_true(not got[3].value())  # "apple" != "banana"


def test_a_string_comparison_over_an_empty_batch() raises:
    """Zero rows is the boundary the loop-per-element string path can get
    wrong where a SIMD path cannot: the offsets buffer of an empty
    `StringArray` still holds one entry, so a length-driven loop is the only
    thing keeping this from reading it."""
    var empty = List[Optional[String]]()
    var a = Int64Builder(0)
    var b = record_batch(
        [array(empty).to_dyn(), a.finish().to_dyn()], names=["name", "a"]
    )
    var got = (
        (col("name", string) > lit("b", string))
        .evaluate(b.to_struct_array(), Bindings())
        .to_array(0)
        .as_bool()
        .copy()
    )
    assert_equal(len(got), 0)

    var plan = table(b^).filter(col("name", string) > lit("b", string))
    assert_equal(plan.execute().num_rows(), 0)


# ---------------------------------------------------------------------------
# StringUnary — the fused half
# ---------------------------------------------------------------------------
def test_string_upper_transforms_values_not_validity() raises:
    """A map changes what a value is, never whether there is one:
    `upper(null)` is null, not `""`."""
    var b = _batch()
    var got = (
        Upper(col("name", string))
        .evaluate(b.to_struct_array(), Bindings())
        .to_array(4)
    )
    assert_equal(String(got.as_string()[0].value()), "PEAR")
    assert_equal(String(got.as_string()[1].value()), "QUINCE")
    assert_true(got.is_null(2))
    assert_equal(got.null_count(), 1)


def test_string_lower_and_strip_compose_without_materialising() raises:
    """`Strip(Lower(col)) == lit` is one builder pass — the intermediate
    `lower(col)` never becomes a column. Observable as the right answer with
    the operands nested, which is what this pins."""
    var pad: List[Optional[String]] = ["  PeAr  ", "  Quince"]
    var b = record_batch([array(pad).to_dyn()], names=["name"])
    var got = (
        (Strip(Lower(col("name", string))) == lit("pear", string))
        .evaluate(b.to_struct_array(), Bindings())
        .to_array(2)
        .as_bool()
        .copy()
    )
    assert_true(got[0].value())
    assert_true(not got[1].value())


# ---------------------------------------------------------------------------
# StringLength — a breaker producing int32
# ---------------------------------------------------------------------------
def test_string_length_counts_bytes() raises:
    var b = _batch()
    var got = (
        StringLength(col("name", string))
        .evaluate(b.to_struct_array(), Bindings())
        .to_array(4)
        .as_int32()
        .copy()
    )
    assert_equal(got[0].value(), 4)  # pear
    assert_equal(got[1].value(), 6)  # quince
    assert_equal(got[3].value(), 5)  # apple


def test_string_length_of_a_null_is_null() raises:
    """Validity is read off the bound the kernel produced — `ColumnBound` —
    rather than recomputed from the operand."""
    var b = _batch()
    var got = (
        StringLength(col("name", string))
        .evaluate(b.to_struct_array(), Bindings())
        .to_array(4)
    )
    assert_true(got.is_null(2))
    assert_equal(got.null_count(), 1)


def test_string_length_feeds_a_numeric_comparison() raises:
    """The breaker still composes upward: its result is an ordinary numeric
    node, so a comparison over it fuses as any other would."""
    var b = _batch()
    var got = (
        (StringLength(col("name", string)) > lit(4, int32))
        .evaluate(b.to_struct_array(), Bindings())
        .to_array(4)
        .as_bool()
        .copy()
    )
    assert_true(not got[0].value())  # 4 > 4
    assert_true(got[1].value())  # 6 > 4
    assert_true(got.is_null(2))
    assert_true(got[3].value())  # 5 > 4


# ---------------------------------------------------------------------------
# StringPredicate — the breaker, and its scalar-pattern branch
# ---------------------------------------------------------------------------
def test_string_starts_with_a_literal_pattern() raises:
    var b = _batch()
    var got = (
        StartsWith(col("name", string), lit("p", string))
        .evaluate(b.to_struct_array(), Bindings())
        .to_array(4)
        .as_bool()
        .copy()
    )
    assert_true(got[0].value())  # pear
    assert_true(not got[1].value())  # quince
    assert_true(got.is_null(2))
    assert_true(not got[3].value())  # apple


def test_string_ends_with_a_literal_pattern() raises:
    var b = _batch()
    var got = (
        EndsWith(col("name", string), lit("e", string))
        .evaluate(b.to_struct_array(), Bindings())
        .to_array(4)
        .as_bool()
        .copy()
    )
    assert_true(not got[0].value())  # pear ends in 'r'
    assert_true(got[1].value())  # quince
    assert_true(got[3].value())  # apple


def test_string_like_matches_wildcards() raises:
    """`%` spans, `_` is one character. This is the scalar branch, so the
    pattern is compiled once by `LikeKernel.apply_scalar` rather than per row.
    """
    var b = _batch()

    def matches(var pattern: String) raises {imm} -> List[Bool]:
        var got = (
            Like(col("name", string), lit(pattern^, string))
            .evaluate(b.to_struct_array(), Bindings())
            .to_array(4)
            .as_bool()
            .copy()
        )
        # Row 2 is null in every case; a null never matches.
        assert_true(got.is_null(2))
        return [got[0].value(), got[1].value(), got[3].value()]

    assert_true(matches("p%") == [True, False, False])  # prefix
    assert_true(matches("%n%") == [False, True, False])  # contains
    assert_true(matches("_ear") == [True, False, False])  # one-char wildcard
    assert_true(matches("%e") == [False, True, True])  # suffix


def test_string_ilike_ignores_case() raises:
    var b = _batch()
    var got = (
        ILike(col("name", string), lit("P%", string))
        .evaluate(b.to_struct_array(), Bindings())
        .to_array(4)
        .as_bool()
        .copy()
    )
    assert_true(got[0].value())  # 'pear' matches 'P%'
    assert_true(not got[1].value())


def test_string_like_against_a_column_pattern() raises:
    """The non-scalar branch: a per-row pattern cannot be compiled once, so
    this goes through `apply` instead of `apply_scalar`."""
    var pats: List[Optional[String]] = ["p%", "z%"]
    var names: List[Optional[String]] = ["pear", "quince"]
    var b = record_batch(
        [array(names).to_dyn(), array(pats).to_dyn()],
        names=["name", "pat"],
    )
    var got = (
        Like(col("name", string), col("pat", string))
        .evaluate(b.to_struct_array(), Bindings())
        .to_array(2)
        .as_bool()
        .copy()
    )
    assert_true(got[0].value())  # 'pear' LIKE 'p%'
    assert_true(not got[1].value())  # 'quince' LIKE 'z%'


def test_string_predicate_feeds_a_filter() raises:
    """The output is bit-packed exactly as a numeric comparison's is, which is
    what lets a breaker predicate drive `Filter` unchanged."""
    var b = _batch()
    var plan = table(b^).filter(
        StartsWith(col("name", string), lit("p", string))
    )
    assert_equal(plan.execute().num_rows(), 1)
