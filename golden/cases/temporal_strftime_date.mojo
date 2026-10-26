# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT strftime(d, '%d/%m/%Y %H:%M') AS s FROM events

    A `date32` formats as its midnight: the clock fields are zero.

    -- skip python

    -- expected
    s:string
    '01/01/2021 00:00'
    '15/06/2021 00:00'
    NULL
    '29/02/2020 00:00'
    '31/12/2021 00:00'
    '15/06/2021 00:00'
    """
    var t = table("events")
    return t.project(
        ["s"], [strftime(col("d"), lit(StringScalar("%d/%m/%Y %H:%M")))]
    )
