from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT upper(s[1:1]) || lower(s[2:]) AS c FROM words

    DuckDB has neither `capitalize` nor `initcap`, so the twin spells the
    operation out: first character upper-cased, the rest lower-cased -- pyarrow's
    `utf8_capitalize`, which is what marrow implements.

    `  pad  ` is the case that separates this from a per-word `initcap`: the
    first character is a space, so nothing is upper-cased at all. `héllo` checks
    that the split is by character and not by byte.

    -- expected
    c:string
    'Hello'
    'World'
    '  pad  '
    ''
    'Héllo'
    NULL
    """
    var t = table("words")
    var q = t.project(["c"], [col("s", string).capitalize()])
    return q
