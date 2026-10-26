# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT to_hex(from_hex(lower(to_hex(s)))) AS h FROM words

    Decoding takes either case: the digits are lower-cased before they are
    decoded, and encoding them again gives the upper-case spelling back.

    -- skip python

    -- expected
    h:string
    '48656C6C6F'
    '774F524C44'
    '20207061642020'
    ''
    '68C3A96C6C6F'
    NULL
    """
    var t = table("words")
    return t.project(["h"], [to_hex(from_hex(lower(to_hex(col("s")))))])
