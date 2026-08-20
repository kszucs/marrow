from golden.helpers import table
from marrow.dtypes import string
from marrow.expr.builders import col
from marrow.expr.relations import DynRelation
from marrow.expr.values import Upper


def plan() raises -> DynRelation:
    """
    SELECT upper(s) AS u FROM words

    -- expected
    u:string
    'HELLO'
    'WORLD'
    '  PAD  '
    ''
    'HÉLLO'
    NULL
    """
    var t = table("words")
    var q = t.project(["u"], [Upper(col("s", string))])
    return q
