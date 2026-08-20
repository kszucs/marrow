from golden.helpers import table
from marrow.dtypes import string
from marrow.expr.builders import col
from marrow.expr.relations import DynRelation
from marrow.expr.values import StringLength


def plan() raises -> DynRelation:
    """
    SELECT CAST(strlen(s) AS INTEGER) AS n FROM words

    Bytes, not codepoints: `héllo` is 6. The twin says `octet_length`
    deliberately — DuckDB's `length` counts characters, which is the other
    answer and would make this case assert the wrong thing.

    -- expected
    n:int32
    5
    5
    7
    0
    6
    NULL
    """
    var t = table("words")
    var q = t.project(["n"], [StringLength(col("s", string))])
    return q
