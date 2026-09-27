# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT s LIKE 'h%' AS b FROM words

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
    return t.project(["b"], [col("s", string).like(lit("h%", string))])
