"""The string family — the one that cannot vectorise.

`NumericValue.lane[W]` and `BoolValue.lane[W]` answer `W` elements because
their storage is fixed-width. UTF-8 is not, so `StringValue.lane` answers one
`String` and every node here loops. These cases assert that the family still
composes exactly like the others: a string predicate bit-packs, feeds `And`,
and drives a `Filter` unchanged.
"""

from std.testing import assert_equal, assert_true

from ...params import Bindings
from ....builders import array, StringBuilder
from ....arrays import StringArray
from ....dtypes import (
    DynType,
    Int64Type,
    StringType,
    int64,
    string,
)
from ....tabular import RecordBatch, record_batch
from ...logical import DynValue
from ...logical import DynRelation, Filter, InMemoryTable, Project
from ..boolean import And
from ..leaves import Column, Literal, StringColumn, StringLiteral
from ..numeric import Gt
from ..strings import StrEq, StrGt, StrLt, StrNe


def _names() raises -> StringArray:
    """`array()` has no `Optional[String]` overload, so a nullable string
    column is built through the builder."""
    var b = StringBuilder(4)
    b.append("pear")
    b.append("quince")
    b.append_null()
    b.append("apple")
    return b.finish()


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
        StringColumn[StringType]("name").evaluate(b, Bindings()).to_array(4)
    )
    assert_true(got.as_string() == _names())


def test_a_string_literal_stays_scalar() raises:
    var lit = StringLiteral[StringType]("pear")
    assert_true(lit.evaluate(_batch(), Bindings()).is_scalar())
    assert_equal(len(lit.columns()), 0)


def test_string_equality_bit_packs_like_a_numeric_comparison() raises:
    """The property that lets a string predicate feed `And` and `Filter`
    unchanged: the output is a packed bool column, whatever the input width."""
    var b = _batch()
    var pred = StrEq(
        StringColumn[StringType]("name"), StringLiteral[StringType]("pear")
    )
    var got = pred.evaluate(b, Bindings()).to_array(4).as_bool().copy()
    assert_true(got[0].value())
    assert_true(not got[1].value())
    assert_true(got.is_null(2))  # NULL = 'pear' is NULL, not false
    assert_true(not got[3].value())


def test_a_null_string_compares_to_null_not_false() raises:
    """Null-in, null-out. The data bit is computed regardless — the loop
    compares whatever bytes are there — so validity is the only record that the
    answer is meaningless."""
    var b = _batch()
    for pred in [
        StrNe(
            StringColumn[StringType]("name"), StringLiteral[StringType]("pear")
        )
    ]:
        var got = pred.evaluate(b, Bindings()).to_array(4).as_bool().copy()
        assert_true(got.is_null(2))


def test_string_ordering() raises:
    var b = _batch()
    var lt = (
        StrLt(
            StringColumn[StringType]("name"), StringLiteral[StringType]("pear")
        )
        .evaluate(b, Bindings())
        .to_array(4)
        .as_bool()
        .copy()
    )
    assert_true(lt[3].value())  # "apple" < "pear"
    assert_true(not lt[1].value())  # "quince" > "pear"

    var gt = (
        StrGt(
            StringColumn[StringType]("name"), StringLiteral[StringType]("pear")
        )
        .evaluate(b, Bindings())
        .to_array(4)
        .as_bool()
        .copy()
    )
    assert_true(gt[1].value())


def test_a_string_predicate_fuses_with_a_numeric_one() raises:
    """Two families in one fused subtree, which is the whole claim: `lane`
    returns different things, but both bit-pack, so `And` composes them without
    knowing either."""
    var plan = DynRelation(
        Filter(
            DynRelation(InMemoryTable(_batch())),
            DynValue(
                And(
                    StrGt(
                        StringColumn[StringType]("name"),
                        StringLiteral[StringType]("b"),
                    ),
                    Gt(Column[Int64Type]("a"), Literal[Int64Type](1)),
                )
            ),
        )
    )
    var out = plan.execute()
    # name > "b" and a > 1  ->  only "quince" (a=2); the null does not select
    assert_equal(out.num_rows(), 1)
    assert_true(out.columns[1].as_int64() == array([2], int64))


def test_a_string_column_projects() raises:
    var b = _batch()
    var plan = DynRelation(
        Project(
            DynRelation(InMemoryTable(b.copy())),
            ["who"],
            [DynValue(StringColumn[StringType]("name"))],
        )
    )
    assert_true(plan.schema().fields[0].dtype == DynType(string))
    assert_true(plan.execute().columns[0].as_string() == _names())
