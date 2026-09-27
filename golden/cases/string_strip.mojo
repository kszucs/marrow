# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT trim(s) AS p FROM words

    -- expected
    p:string
    'Hello'
    'wORLD'
    'pad'
    ''
    'héllo'
    NULL
    """
    var t = table("words")
    return t.project(["p"], [col("s", string).strip()])
