# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Query datasets on the Hugging Face Hub, without downloading them.

`marrow.datasets.load_dataset` is shaped like the `datasets` function of the
same name. It asks the Hub which files make up a config's split -- from the
dataset card's `configs:` block, or from the file names -- and returns a lazy
table scanning all of them. Only the first file's footer is read to build the
plan; running it fetches just the column chunks the query needs, from every
shard in turn.

A dataset stored as Parquet, JSON Lines or Arrow is read from its own files.
One stored in another format, like the CSV of `scikit-learn/iris`, is read
from the Parquet copy the Hub converts every public dataset into.

Requires `libopendal_c` (a local path never needs it):

    pixi run -e opendal build_opendal

Run with:
    pixi run -e opendal huggingface
"""

import time

from marrow import col
from marrow.datasets import (
    get_dataset_config_names,
    get_dataset_split_names,
    load_dataset,
)


def main() -> None:
    started = time.monotonic()

    # Metadata only: one request to the Hub's API.
    print("gsm8k configs:", get_dataset_config_names("openai/gsm8k"))
    print("gsm8k main splits:", get_dataset_split_names("openai/gsm8k", "main"))

    # Building the plan reads the first file's footer, not its data.
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    print(f"\ncolumns: {gsm.column_names}")

    query = (
        gsm.select("question")
        .filter(col("question").char_length() > 300)
        .limit(5)
    )

    # `.optimize()` is opt-in: `collect()` alone applies no rules, so without
    # it the scan keeps the full schema and `answer`'s column chunks are
    # downloaded and then thrown away. `ColumnPruning` narrows the scan to
    # `question`, and that is what stops the bytes leaving the Hub.
    longest = query.optimize()
    print("\nplan as run:")
    print(longest.explain())

    result = longest.collect()
    print(f"\n{result.num_rows} of the longest questions:\n")
    for i in range(result.num_rows):
        question = result.column("question")[i].as_py()
        print(f"  - {question[:96]}...")

    # A CSV dataset, read from the Hub's Parquet conversion of it.
    iris = load_dataset("scikit-learn/iris").collect()
    print(f"\niris: {iris.num_rows} rows, columns {iris.column_names}")

    print(f"\nelapsed: {time.monotonic() - started:.1f}s over the network")


if __name__ == "__main__":
    main()
