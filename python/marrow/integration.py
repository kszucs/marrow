# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Arrow integration testing: the JSON format `archery` generates.

Test infrastructure for the cross-implementation suite, not a data format.
"""

from . import libmarrow as _ma
from .tabular import RecordBatch
from .types import Schema

__all__ = ["c_data_allocations", "read_json"]


def c_data_allocations():
    """How many heap blocks the C Data structs marrow exported or imported
    still hold: back where it started once each has been released."""
    return _ma.c_data_allocations()


def read_json(path):
    """Read an integration JSON file: ``(schema, batches)``."""
    schema, batches = _ma.read_integration_json(str(path))
    return Schema.wrap(schema), [RecordBatch.wrap(b) for b in batches]
