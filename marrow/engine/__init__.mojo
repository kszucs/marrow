# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""The execution engine: phases, their sources, and the drivers that run
them."""

from .driver import (
    Driver,
    Phase,
    all_blocking,
    append_each,
    rows_bound,
    rows_of,
    runs_in_parallel,
)
from .report import PhaseReport, PlanReport
from .sources import (
    DynSource,
    IpcSource,
    OperatorSource,
    PlannedSource,
    RowGroupSource,
    Source,
    TableSource,
    UnmatchedSource,
)
