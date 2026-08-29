from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT s ILIKE '%É%' AS b FROM words

    Case folding beyond ASCII. `string_ilike_is_not` uses an ASCII pattern,
    which `ILikeKernel` answers on its cheap byte-folding path; a pattern
    holding `É` forces the Unicode one, where the fold has to know that `É` and
    `é` are the same letter. Only `héllo` may match.

    -- expected
    b:bool
    False
    False
    False
    False
    True
    NULL
    """
    var t = table("words")
    return t.project(["b"], [ILike(col("s", string), lit("%É%", string))])
