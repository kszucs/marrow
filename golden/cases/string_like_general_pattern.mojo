from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT s LIKE '%l%o' AS b FROM words

    Two wildcards with literal text between and after them — the backtracking
    arm of `LikeKernel`, which the corpus otherwise only reaches through its
    `foo%`, `%foo` and `%foo%` shortcuts. Matching `Hello` requires trying the
    first `l`, failing to end at `o`, and retrying from the second.

    -- expected
    b:bool
    True
    False
    False
    False
    True
    NULL
    """
    var t = table("words")
    return t.project(["b"], [Like(col("s", string), lit("%l%o", string))])
