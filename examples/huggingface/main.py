"""Query a Parquet file on the Hugging Face Hub, without downloading it.

Nothing here is Hugging Face-specific. The Hub serves Parquet over plain HTTPS
with `Accept-Ranges: bytes`, which is all marrow's `https://` backend needs:
the reader opens a file by parsing its footer out of a bounded tail, then
fetches only the column chunks the query asks for. A 2.3 MB file is small
enough to make that hard to see and big enough to prove it works; the same
code against a multi-gigabyte object fetches the same handful of ranges.

`read_parquet` picks the backend from the URI scheme, so the only difference
from a local path is the string. The query, the optimizer and the execution
engine are unchanged and unaware.

Requires `libopendal_c` (a local path never needs it):

    pixi run -e opendal build_opendal
    pixi run -e opendal python examples/huggingface/main.py

Run with:
    pixi run -e opendal huggingface
"""

import time

import marrow as ma
from marrow import col

# GSM8K's train split, committed as Parquet on the repo's main branch.
#
# `hf://` is the Hub's own scheme, spelled the way `huggingface_hub`'s
# `HfFileSystem` spells it, and it is what OpenDAL's `hf` service takes: the
# repo becomes operator configuration and only the file path is the key.
# `HF_TOKEN` is picked up from the environment for a private repo.
#
# The plain HTTPS URL for the same file also works and needs no `hf` service:
#
#   https://huggingface.co/datasets/openai/gsm8k/resolve/main/main/train-00000-of-00001.parquet
#
# What does *not* work either way is the Hub's auto-converted
# `refs/convert/parquet` branch: naming it needs a revision containing
# slashes, which neither an object key nor an HTTPS path can carry.
URI = "hf://datasets/openai/gsm8k/main/train-00000-of-00001.parquet"


def main() -> None:
    started = time.monotonic()

    # Metadata only: this reads the footer, not the file.
    table = ma.read_parquet(URI)
    print(f"columns: {table.column_names}")

    query = (
        table.select("question")
        .filter(col("question").char_length() > 300)
        .limit(5)
    )

    # `.optimize()` is the whole point, and it is opt-in: `collect()` alone
    # applies no rules, so without this the scan keeps the full schema and
    # `answer`'s column chunks are downloaded and then thrown away. Compare
    # the two plans -- `ColumnPruning` rewrites `ParquetScan` to name only
    # `question`, and that is what stops the bytes leaving the Hub.
    longest = query.optimize()

    print("\nplan as written:")
    print(query.explain())
    print("\nplan as run:")
    print(longest.explain())

    result = longest.collect()
    print(f"\n{result.num_rows} of the longest questions:\n")
    for i in range(result.num_rows):
        question = result.column("question")[i].as_py()
        print(f"  - {question[:96]}...")

    print(f"\nelapsed: {time.monotonic() - started:.1f}s over the network")


if __name__ == "__main__":
    main()
