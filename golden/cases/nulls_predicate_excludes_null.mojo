# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT a, b, g FROM nulls WHERE a > 0

    -- expected
    a:int64	b:int64	g:string
    """
    var t = table("nulls")
    return t.filter(col("a", int64) > lit(0, int64))
