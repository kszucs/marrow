from golden.helpers import table
from marrow.dtypes import string
from marrow.expr.builders import col
from marrow.expr.relations import DynRelation
from marrow.expr.values import Strip


def plan() raises -> DynRelation:
    """
    SELECT trim(s) AS p FROM words

    -- expected
    p:string
    'Hello'
    'wORLD'
    'pad'
    ''
    'héllo'
    NULL
    """
    var t = table("words")
    var q = t.project(["p"], [Strip(col("s", string))])
    return q
