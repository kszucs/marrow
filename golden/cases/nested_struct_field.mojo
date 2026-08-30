from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT st.a AS a, st.b AS b FROM nested ORDER BY id NULLS FIRST

    Reading a struct's fields. The two null rows are why the fixture holds both
    shapes: row 2 has a null *field* inside a valid struct, row 3 is a null
    struct — and both project to null, so the distinction survives only in the
    struct column itself.

    There is no struct leaf: `col` has numeric, string, bool, temporal and list
    overloads.

    -- skip mojo

    -- expected
    a:int64	b:string
    1	'x'
    NULL	'y'
    NULL	NULL
    """
    var t = table("nested")
    var fields = t.project(
        ["a", "b"],
        [
            col("st", struct_of(int64, string)).field("a"),
            col("st", struct_of(int64, string)).field("b"),
        ],
    )
    return fields.sort_by([col("id", int64)], [True])
