<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.svg">
  <img alt="marrow — Apache Arrow in Mojo" src="docs/assets/logo-light.svg" width="560">
</picture>

# marrow

**Apache Arrow in [Mojo](https://www.modular.com/mojo)** — the columnar format,
compute kernels, a Parquet and Arrow IPC layer, a relational query engine, and
Python bindings.

The format is the standard every Arrow library shares. What marrow adds is a
choice of how to use it, from calling a kernel on an array to compiling a whole
query into an executable:

| | You write | You get |
|---|---|---|
| **Eager Python** | `ma.array`, `ma.compute.add`, `rb.sort_by` | the PyArrow API you already know |
| **Lazy Python** | `read_parquet(...).filter(...).aggregate(...)` | nothing runs until `.collect()` |
| **Compiled Mojo** | the same verbs, dtypes fixed at compile time | a native executable that runs without Python |

📖 **[Full documentation → marrow.kszucs.dev](https://marrow.kszucs.dev)**

> **Status: experimental.** Both Arrow and Mojo are moving targets. Of 278 golden
> SQL queries whose answers come from **DuckDB**, never from marrow, 215 run and
> match and 63 need features marrow does not have yet. Arrow's integration suite round-trips data with the **C++, Rust and
> Go** implementations for the layouts marrow implements. See
> [Status & limitations](https://marrow.kszucs.dev/reference/status.html) for
> what is missing and what is known to be wrong.

## Install

marrow is a Mojo project with a Python extension. Build it with
[pixi](https://pixi.sh):

```bash
git clone https://github.com/kszucs/marrow && cd marrow
pixi run build_python          # compiles python/marrow/libmarrow.so
```

## Sixty seconds

```python
import marrow as ma
from marrow import col, lit

# Eager — PyArrow shapes, with type inference and null support
a = ma.array([1, 2, 3, None, 5])
s = ma.array(["hello", None, "world"])
print(ma.compute.add(a, ma.array([10, 20, 30, 40, 50])))

# Lazy — nothing runs until collect()
batch = ma.record_batch({
    "region": ma.array(["east", "west", "east"]),
    "price":  ma.array([10, 20, 30]),
})
print(
    ma.memtable(batch)
      .filter(col("price") > lit(15))
      .aggregate(by=["region"], total=("sum", "price"))
      .collect()
      .to_pylist()
)
```

Arrays move to and from PyArrow over the Arrow C Data Interface without
copying their buffers:

```python
import pyarrow as pa
pa_arr = pa.array(ma.array([1, 2, 3]))     # marrow -> PyArrow, no copy
ma_arr = ma.array(pa.array([1, 2, 3]))     # PyArrow -> marrow, no copy
```

## Compiled queries

A query written against the Mojo expression layer compiles to a native
executable that needs no Python at run time; only scalars and paths are
supplied when it runs. It still loads the Mojo runtime libraries, which
`--bundle` copies alongside it:

```mojo
from marrow.dtypes import field, int64, string
from marrow.expr import QueryCli, col, scan
from marrow.schema import schema

def main() raises:
    var cli = QueryCli("orders", description="Orders above a threshold.")
    var min_amount = cli.param("min-amount", int64, default=Int64(0))
    cli.argument("src", help="input Parquet file")

    if cli.parse():
        var sch = schema([field("id", int64), field("amount", int64)])
        cli.run(
            scan(cli.get("src"), sch^).filter(col("amount", int64) >= min_amount)
        )
```

```bash
marrow compile query.mojo -o orders
./orders orders.parquet --min-amount 250
./orders --help          # generated from the param() declarations
```

`--help` and `--describe` are rendered from the declarations themselves, so
there is no argument-parsing code to keep in sync.
See the [compile guide](https://marrow.kszucs.dev/guide/compile.html).

## What's in it

- **Layouts** — bool, numeric, string/binary (+large), fixed-size binary,
  list/large_list/fixed_size_list, struct, map, dictionary, decimal
  (32/64/128/256) and the temporal family. Union, run-end-encoded and view
  layouts are not implemented.
- **Kernels** — arithmetic, comparison, boolean, cast, aggregate, distinct,
  filter/take/drop_null, sort, hash join (6 kinds), group-by, window, string
  (incl. `LIKE`/`ILIKE`), temporal, conditional, membership and nested.
- **Query engine** — a push-based executor, a 16-rule optimizer with column
  pruning, statistics-based Parquet pruning, and late-bound parameters.
- **I/O** — a from-scratch Parquet reader and writer (snappy/zstd/lz4, page v1
  and v2, statistics, page index) with no PyArrow at runtime, plus Arrow IPC
  file and stream round-trips.
- **Interop** — the Arrow C Data Interface, release callbacks included.
- **GPU** — element-wise kernels can dispatch to Metal or CUDA from the same
  source as the CPU path, behind `-D MARROW_GPU=true`.

## Development

```bash
pixi run -e dev test                   # everything
pixi run -e dev pytest marrow/kernels/tests/test_join.mojo   # one file
pixi run -e dev precompile             # fast compile check, no test run
pixi run -e dev fmt                    # mojo format + ruff
pixi run -e docs docs                  # build the documentation site
pixi run binary_size                   # the AOT binary-size gate
```

Contributions welcome. `CLAUDE.md` carries the architecture, the coding rules
and the compiler gotchas; `backlog.md` carries the open work.

## References

- [Arrow columnar format](https://arrow.apache.org/docs/format/Columnar.html)
- [Arrow C Data Interface](https://arrow.apache.org/docs/format/CDataInterface.html)
- [arrow.mojo](https://github.com/mojo-data/arrow.mojo) — another Arrow-in-Mojo effort

## License

Apache 2.0 — see [LICENSE.txt](LICENSE.txt).
