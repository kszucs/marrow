# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""A Hub dataset's split and a glob of local files, each as a plan.

Compiled by `pixi run docs_check`; included by docs/guide/datasets.qmd.
"""

from marrow.datasets import DataFiles, HubDataset, load_dataset
from marrow.dtypes import int64
from marrow.expr import col, lit


def main() raises:
    # What the Hub says about a dataset: its configs, and each one's splits.
    var hub = HubDataset.fetch("openai/gsm8k")
    print(hub)

    # One split of one config, as a scan over every file in it.
    var gsm = load_dataset("openai/gsm8k", "main", split="test")
    print(gsm.limit(3).execute())

    # Files of one format, named by a glob.
    var orders = load_dataset(
        "parquet", DataFiles(["data/orders-*.parquet"])
    ).filter(col("amount", int64) > lit(100, int64))
    print(orders.execute())
