# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""The execution engine: operators and their pipelines, phases, their sources,
and the drivers that run them. It knows no plan: a plan lowers itself into
phases (`marrow.expr`)."""

from .core import Datum, DynOperator, Morsel, Operator, Pipeline
from .driver import (
    MORSEL_ROWS,
    Driver,
    Phase,
    append_each,
    runs_in_parallel,
)
from .report import PhaseReport, PhaseRun, PlanProfile, PlanReport
from .sources import (
    DynSource,
    Source,
    Stream,
    StreamSource,
    TableSource,
    rows_of,
)
