# Organizational audit

Where code lives versus where its responsibility says it belongs: package layout,
module boundaries, public surface, and the import edges that cross a layer.

Verified against the working tree at `f5226d5` + uncommitted changes
(2026-08-16). **Nothing here has been changed** — this is a findings list.

Companion to `docs/abstraction-audit.md` (trait hierarchies and leaks) and
`docs/duplication-audit.md` (repeated code). Three items are shared with the
abstraction audit because a misplaced *type* and a leaky *abstraction* are the
same defect seen from two sides; they are cross-referenced, not repeated.

---

## 0. The shape today

| | files | source lines |
|---|---|---|
| `marrow/` root | 12 | 16,367 |
| `marrow/kernels/` | 21 | 11,314 |
| `marrow/parquet/` | 10 | 8,546 |
| `marrow/expr/` | 7 | 6,435 |
| `marrow/testing/` | 3 | 614 |

The root's 12 modules cover three unrelated concerns:

| concern | modules | lines |
|---|---|---|
| **data model** | `dtypes`, `buffers`, `views`, `arrays`, `builders`, `scalars`, `schema`, `tabular` | 11,591 |
| **interchange** | `ipc`, `c_data` | 4,105 |
| **policy / util** | `execution`, `utils` | 671 |

---

## 1. Findings

### 1.1 `marrow/__init__.mojo` is empty — 0 bytes

Every subpackage has a substantial one: `kernels` 83 lines, `expr` 75,
`parquet` 49, `testing` 4. The root has no docstring, no re-exports, no statement
of what the library's public surface is.

`parquet/__init__.mojo` opens with `from marrow.parquet import read_table,
write_table` and means it. There is no equivalent sentence anywhere for `marrow`
itself — callers reach into `marrow.arrays`, `marrow.dtypes`, `marrow.kernels`
individually, and nothing distinguishes the intended API from the internals.

This is also why the `Known Limitations` and architecture sections of `CLAUDE.md`
have to carry the map: the package cannot.

### 1.2 `kernels/__init__.mojo` promises re-exports for 17 modules and delivers 8

The docstring says callers can `import marrow.kernels as mk` and use
`mk.AddKernel.dispatch`, `mk.SumKernel.dispatch`, `mk.filter`, `mk.sort`
"directly", then lists 17 submodules.

Actually re-exported: `aggregate`, `numeric`, `cast`, `distinct`, `filter`,
`membership`, `sort`, plus `ExecContext` from `execution`.

Not re-exported: `boolean`, `string`, `temporal`, `conditional`, `nested`,
`concat`, `groupby`, `join`, `hashing`, `hashtable`, `partition`, `interval`,
`core`.

So `mk.cast` and `mk.sort` work while `mk.upper`, `mk.concat`, `mk.coalesce` and
`mk.is_null` do not, and no principle separates the two lists — `concat` and
`cast` are equally top-level operations. A caller discovers the boundary by
hitting it.

### 1.3 Two absolute `from marrow.*` imports in library code

- [views.mojo:20](marrow/views.mojo#L20) — `from marrow.utils import has_accelerator_support`
- [kernels/__init__.mojo:31](marrow/kernels/__init__.mojo#L31) — `from marrow.dtypes import (…)`

Everything else in the tree is relative. `CLAUDE.md` states the rule as absolute:
*"Absolute `from marrow.x import` fails with `unable to locate module 'marrow'`
when compiled as part of the package."* Both of these compile today, so either the
rule is narrower than written (it is stated in the **tests** section and may only
bind there) or these two are latent breakage. Worth resolving in one direction —
the rule as written and the code as written disagree.

Note the first one disappears entirely under §1.5.

### 1.4 Parquet is a package; IPC and C-Data are loose root modules

All three are interchange formats. Parquet gets 10 modules with a clean internal
split. IPC gets one 2,425-line file — which **already has the same seams, labelled**:

| `ipc.mojo` section | lines | `parquet/` counterpart |
|---|---|---|
| wire constants, internal structs | 38–251 | `format.mojo` |
| generic FlatBuffers codec | 252–629 | `format.mojo` (Thrift codec) |
| metadata encoder | 630–1228 | `schema.mojo` + `writer.mojo` |
| metadata decoder | 1229–1623 | `schema.mojo` + `reader.mojo` |
| framing / message reader | 1624–1705 | `source.mojo` |
| batch body encoder | 1706–1875 | `writer.mojo` |
| batch body decoder | 1876–2064 | `reader.mojo` |
| public writers / readers / functions | 2065–2423 | `__init__.mojo` |

`c_data.mojo` (1,680 lines) is the third member of the group and also sits loose
at the root, next to `arrays.mojo` and `buffers.mojo`.

Grouping the three under `marrow/io/` (or `formats/`) would leave the root as the
data model and nothing else. It would also give **abstraction audit §1.8** — the
plan layer's hard dependency on `marrow.parquet` — a natural seam to break
against: `expr` would depend on an `io` boundary rather than on one format's
reader.

### 1.5 `utils.mojo` is four unrelated things with four disjoint consumer sets

333 lines, no single responsibility. The consumer sets do not overlap at all:

| group | members | consumers | natural home |
|---|---|---|---|
| comptime dispatch | `narrow`, `variant_dispatch`, `variant_dispatch_raises` | `dtypes`, `arrays`, `builders`, `scalars` + `kernels/{temporal,aggregate,string}` | stays (or `dtypes.mojo`, its heaviest user) |
| device policy | `GPU_ENABLED`, `has_accelerator_support` | `views` + `kernels/{boolean,hashing,cast,numeric}` | **`execution.mojo`** |
| byte order | `LittleEndian` | `ipc` + all 6 `parquet` modules | **the interchange group (§1.4)** |
| checksums | `Crc32` | `parquet/{reader,writer,bloom}` only | **`parquet/`** |

`LittleEndian` and `Crc32` have **no consumer outside the interchange formats** —
they are Parquet/IPC implementation details living in a core module that every
part of the tree imports.

Moving device policy to `execution.mojo` is the highest-value of the four:
`execution.mojo` is already *the* policy module ("it imports nothing from
marrow — it is a pure policy object"), `views.mojo` already imports it, and doing
so deletes the absolute import flagged in §1.3.

There is precedent for exactly this move: `execution.mojo`'s own docstring records
that it used to live under `kernels/` and was moved out because *"filing it under
`kernels/` was the tree's only `core -> kernels` import edge"*.

### 1.6 `kernels/` holds four things that are not kernels

- `hashtable.mojo` — `SwissHashTable`, a data structure
- `partition.mojo` — `RadixPartitioner`, a data structure plus the shared
  hash → partition → parallel → merge skeleton
- `hashing.mojo` — `rapidhash`, a hash function
- `core.mojo` — holds `Grouping`, a two-field data type, in the module whose
  docstring calls itself "the root of the kernel trait hierarchy"

None of the first three conforms to `Kernel`; all exist to serve `groupby`,
`join`, `distinct` and `membership`. The `__init__` docstring is candid about it
("the hash machinery group-by, join and `is_in` share"), which is the tell: the
package name no longer describes its contents.

Low severity — the grouping is by *client* rather than by *kind*, and that is a
defensible axis. Recorded so "everything in `kernels/` conforms to `Kernel`" is
not assumed by a future change.

### 1.7 `tabular.mojo` reaches up into `expr` — the one cross-layer cycle

[tabular.mojo:23](marrow/tabular.mojo#L23) — `from .expr.aggregates import FoldedAggregates`.

Full treatment in **abstraction audit §1.7**. Organizationally: this is the only
import edge in the tree that goes from a lower layer to a higher one. Every other
cycle (`arrays ↔ scalars`, `arrays ↔ builders`, `buffers ↔ views`,
`expr.values ↔ expr.relations ↔ expr.dynamic`) is *inside* one package, which
Mojo resolves and which `CLAUDE.md` explicitly says not to reorganize around.

The misplaced piece is the aggregate **catalog**: `kernels/aggregate.mojo` owns
the resolution contract (`AggFunction` — "a name plus the input dtypes it
supports") but its catalog (`Sum`, `Min`, `Count`, …) and the `FoldedAggregates`
fold live in `expr/aggregates.mojo`. Nothing about "the aggregate named `sum` over
an int64 column" is an expression concept, and both `tabular` and `expr` consume
it.

### 1.8 `equal_any` creates the `kernels.numeric → kernels.string` edge

[numeric.mojo:583](marrow/kernels/numeric.mojo#L583) is the one place a runtime
dtype selects between the numeric and string comparison families. It is
well-justified and well-documented — two callers (hash-join row verification,
`nullif`) need equality as a primitive over an arbitrary dtype.

Its *placement* is what creates the edge: `numeric.mojo` imports `string.mojo`
solely for `StringEqKernel`, which sits oddly against
`NumericCompareKernel`'s docstring three hundred lines above insisting that
comparing strings is a different family and that the kernel deliberately knows
nothing about it. A neutral home — `kernels/compare.mojo`, or the package
`__init__` — removes the edge without touching the function.

### 1.9 `interval.mojo` — examined and upheld

`kernels/interval.mojo` is consumed **exclusively** by `expr/`
(`pruning`, `values`, `dynamic`, `relations`) and never touches an array. That
shape usually means "misfiled", so: its placement is argued at length in the
module docstring, and the argument holds — an interval reading of an operator is a
peer of its SIMD reading, and belongs beside it, by the same logic that removed
`NumericCompareKernel`'s `comptime StringKernel`.

Recorded here so it is not re-opened. The one real consequence is noted in
abstraction audit §1.6: it is the only `kernels/` module that never sees an array,
which is why `IntervalKernel(Kernel)` inherits `expect_same_length` /
`expect_same_dtype` it can never call.

### 1.10 File sizes

`arrays` 2,866 · `expr/values` 2,763 · `parquet/reader` 2,542 · `ipc` 2,425 ·
`views` 2,076 · `builders` 1,943 · `c_data` 1,680 · `dtypes` 1,569 ·
`buffers` 1,435 · `parquet/format` 1,416 · `parquet/schema` 1,322.

**There is no build-time argument for splitting these.** A test selection
elaborates all of marrow it imports, and everything imports `arrays`; splitting a
file does not shrink the compilation unit. The argument is navigation only, and
`CLAUDE.md` warns against reorganizing to dodge circular imports — so a split
needs a real seam.

`ipc.mojo` is the one file where the seam is already drawn and labelled (§1.4).
`views.mojo` is the second candidate: `BufferView` (100–464) and `BitmapView`
(465–1252) are independent of the 13 `apply` overloads and 5 dispatch drivers
(1253–2076), and the latter is a coherent unit — it is the CPU/GPU execution
driver, which is a different responsibility from "a borrowed, offset-applied
span". The rest are large but cohesive.

---

## 2. Summary

| # | Finding | Severity |
|---|---|---|
| 1.5 | `utils.mojo` has four disjoint consumer sets; `LittleEndian`/`Crc32` are format details in a core module | **high** — clearest single fix |
| 1.7 | `tabular → expr`, the only cross-layer import edge; aggregate catalog misplaced | **high** — see abstraction §1.7 |
| 1.4 | Parquet is a package, IPC (2.4k lines, seams already labelled) and C-Data are loose root modules | **medium** |
| 1.2 | `kernels/__init__` re-exports 8 of 21 modules while documenting 17 | **medium** — user-visible |
| 1.1 | `marrow/__init__.mojo` is empty; no stated public surface | **medium** |
| 1.8 | `equal_any`'s placement creates `numeric → string` | **low** |
| 1.3 | Two absolute `from marrow.*` imports contradict the documented rule | **low** — one disappears under 1.5 |
| 1.6 | `kernels/` holds 3 data structures and a data type | **low** — grouped by client, defensibly |
| 1.10 | Two files with real seams (`ipc`, `views`); no build-time argument | **low** |
| 1.9 | `interval.mojo` placement — examined, upheld | none |
