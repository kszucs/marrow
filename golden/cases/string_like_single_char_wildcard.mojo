from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT s LIKE '_ello' AS b FROM words

    `_` matches exactly one character, and the question is *character* or
    *byte*: `héllo` is five characters and six bytes, and its second character
    is not `e`, so it must not match while `Hello` does. A byte-wise `_` would
    make the two strings differ in length and answer differently.

    The pattern has no `%`, so it takes `LikeKernel`'s general token path
    rather than the prefix, suffix or contains fast cases the other LIKE cases
    reach.

    -- expected
    b:bool
    True
    False
    False
    False
    False
    NULL
    """
    var t = table("words")
    return t.project(["b"], [Like(col("s", string), lit("_ello", string))])
