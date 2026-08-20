from golden.helpers import table
from marrow.dtypes import Int64Type, string
from marrow.expr.builders import col
from marrow.expr.relations import DynRelation
from marrow.expr.values import StringToNum


def plan() raises -> DynRelation:
    """
    SELECT TRY_CAST(s AS BIGINT) AS c FROM nums

    `TRY_CAST`, because 'abc' does not parse: DuckDB's plain CAST raises and
    marrow nulls the value, so the twin has to ask DuckDB the same question
    marrow answers.

    -- expected
    c:int64
    1
    -2
    NULL
    NULL
    """
    var t = table("nums")
    var q = t.project(["c"], [StringToNum[Int64Type](col("s", string))])
    return q
