# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT strftime(ts, '%A, %d %B %Y %I:%M %p') AS s, strftime(ts, '%j %a %b %y') AS t FROM events

    Weekday and month names, full and abbreviated, the 12-hour clock with
    midnight as `12 AM`, and the day of the year (`060` on the leap day).

    -- skip python

    -- expected
    s:string	t:string
    'Friday, 01 January 2021 12:00 AM'	'001 Fri Jan 21'
    'Tuesday, 15 June 2021 12:30 PM'	'166 Tue Jun 21'
    'Tuesday, 15 June 2021 12:30 PM'	'166 Tue Jun 21'
    NULL	NULL
    'Saturday, 29 February 2020 11:59 PM'	'060 Sat Feb 20'
    'Friday, 31 December 2021 11:59 PM'	'365 Fri Dec 21'
    """
    var t = table("events")
    return t.project(
        ["s", "t"],
        [
            strftime(col("ts"), lit(StringScalar("%A, %d %B %Y %I:%M %p"))),
            strftime(col("ts"), lit(StringScalar("%j %a %b %y"))),
        ],
    )
