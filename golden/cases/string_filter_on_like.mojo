from golden.helpers import table
from marrow.dtypes import string
from marrow.expr.builders import col, lit
from marrow.expr.relations import DynRelation
from marrow.expr.values import Like


def plan() raises -> DynRelation:
    """
    SELECT s FROM words WHERE s LIKE '%o%'

    -- expected
    s:string
    'Hello'
    'héllo'
    """
    var t = table("words")
    var q = t.filter(Like(col("s", string), lit("%o%")))
    return q
