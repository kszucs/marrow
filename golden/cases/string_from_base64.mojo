# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT to_base64(from_base64(to_base64(from_hex(to_hex(s))))) AS b FROM words

    Decoding is the inverse of encoding, including across the padding.

    -- skip python

    -- expected
    b:string
    'SGVsbG8='
    'd09STEQ='
    'ICBwYWQgIA=='
    ''
    'aMOpbGxv'
    NULL
    """
    var t = table("words")
    return t.project(
        ["b"], [to_base64(from_base64(to_base64(from_hex(to_hex(col("s"))))))]
    )
