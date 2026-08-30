from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT s, (s = '') AS b FROM words

    The empty string is a value: it compares equal to `''` and unequal to
    everything else, while the null row stays null. Arrow stores both as a
    zero-length slice, so an implementation that reads emptiness off the
    offsets without consulting validity answers true for the null.

    -- expected
    s:string	b:bool
    'Hello'	False
    'wORLD'	False
    '  pad  '	False
    ''	True
    'héllo'	False
    NULL	NULL
    """
    var t = table("words")
    return t.project(
        ["s", "b"], [col("s", string), col("s", string) == lit("", string)]
    )
