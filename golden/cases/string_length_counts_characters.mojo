from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT CAST(length(t) AS BIGINT) AS n FROM text

    The *character* length, which is a different function from the byte length
    `string_length_counts_bytes` asserts: `héllo wörld` is 11 characters and 13
    bytes. DuckDB spells them `length` and `octet_length`; marrow's
    `LengthKernel` implements only the second, so `length` has no equivalent at
    all.

    Recorded here rather than as an `xfail` on the existing case because the
    two are genuinely different functions, and marrow's answer to the one it
    has is correct.

    -- skip python

    -- expected
    n:int64
    5
    3
    0
    NULL
    6
    11
    """
    var t = table("text")
    return t.project(["n"], [col("t", string).char_length()])
