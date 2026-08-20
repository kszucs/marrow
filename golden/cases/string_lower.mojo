from golden.helpers import table
from marrow.dtypes import string
from marrow.expr.builders import col
from marrow.expr.relations import DynRelation
from marrow.expr.values import Lower


def plan() raises -> DynRelation:
    """
    SELECT lower(s) AS l FROM words

    -- expected
    l:string
    'hello'
    'world'
    '  pad  '
    ''
    'héllo'
    NULL
    """
    var t = table("words")
    var q = t.project(["l"], [Lower(col("s", string))])
    return q
