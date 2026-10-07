# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Datasets as plans: `load_dataset` resolves a dataset's split to its files
and returns a scan over them, which runs like any other `DynRelation`.

Explicit re-exports, never `import *`.
"""

from .files import DataFiles
from .formats import FileFormat
from .huggingface import HubConfig, HubDataset
from .load import load_dataset
