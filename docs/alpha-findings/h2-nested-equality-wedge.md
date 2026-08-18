# Why `marrow/tests/test_arrays.mojo` cannot be compiled

**165 tests, none of them runnable.** Six attempts, every one sitting at 0% CPU
until the harness deadline and reporting `165 failed in 1800.0s` with empty
messages. That reads as a harness capacity problem. It is not one — and
`mojo precompile marrow` stays clean throughout, because the *library* builds
fine. Only the test's instantiations do not.

## Bisection

Ranges of the file, run as their own compilation units:

| tests | result |
|---|---|
| 1–40 | 40 passed in 26.0 s |
| 41–80 | 40 passed in 22.6 s |
| 81–90 | 10 passed in 3.4 s |
| **91–100** | **10 failed at the deadline** |
| 101–120 | 20 failed at the deadline |

Individually, at a 120 s deadline:

| test | result |
|---|---|
| `test_string_array_eq_sliced` | **passed in 3.2 s** |
| `test_list_array_eq` | **failed — never compiled** |
| `test_struct_array_eq` | **failed — never compiled** |

So it is not size, and not cumulative: it is *which* tests. Flat-array
equality compiles in seconds; **nested-array equality never compiles at all.**

## Root cause: equality recurses through the erased box

`DynArray.__eq__` (`marrow/arrays.mojo:2540`) is:

```mojo
def __eq__(self, other: DynArray) -> Bool:
    return self._v == other._v
```

`Variant.__eq__` resolves the active member on **both** sides, so this alone
elaborates the box's dispatch ladder squared. That much was already known —
one direct `assert_false(a.to_dyn() == b.to_dyn())` in
`test_temporal_array_dtype_mismatch` hung the compiler on its own; rewriting
that single assertion to compare dtypes made the test build in **3 seconds**.

The nested case is worse, and is why the rest of the file still will not build:

- `ListLikeArray.__eq__` (`:1260`) compares elements with
  `self.unsafe_get(i) != other.unsafe_get(i)`, and `ListLikeArray.unsafe_get`
  (`:1174`) **returns a `DynArray`**.
- `StructArray.__eq__` (`:1939`) compares `children`, which are `DynArray`s.

So `ListLikeArray.__eq__` calls `DynArray.__eq__`, which dispatches over every
member of the variant — **including `ListLikeArray`** — which calls
`ListLikeArray.__eq__` again. The instantiation is structurally recursive and
there is nothing cheap for the compiler to terminate on.

This is the same family CLAUDE.md already warns about under "Associated types,
traits, reflection": *keep recursive and nested ops out of kernel structs — a
binding-compiler crash was once observed on mutually-recursive nested-type
static methods.* Nested equality through an erased box is that shape.

## What was tried and rejected

Rewriting `DynArray.__eq__` to narrow **once** and read the other side at that
same type — O(n) arms instead of O(n²), and exactly as strict, since a type
mismatch is inequality and `isa[T]()` answers it before touching data:

```mojo
def f[T: Array](a: T) {imm} -> Bool:
    if not other._v.isa[T]():
        return False
    return a == other._v[T]
return self._dispatch(f)
```

It compiles and is semantically identical, but **it does not fix this**: the
recursion is through `ListLikeArray.__eq__`, not through the squared ladder, so
`test_list_array_eq` still fails to build. Reverted rather than left in as an
unmeasured change to a hot, size-gated file.

## What would actually fix it

Break the recursion so nested equality does not re-enter the typed ladder.
Candidates, none attempted:

- Compare nested arrays through their flat `ArrayData` (dtype, length, null
  count, bitmap, buffers, children) rather than by recursing into typed
  element equality. Equality of two `ListArray`s is equality of their offsets
  and of their child arrays as buffers.
- Give `DynArray` a non-recursive structural comparison and have the nested
  types call *that* instead of `!=` on the element.

Either is a real design change to a size-gated file and wants its own
measurement, which is why this is written up rather than done.

## What was changed

Only `test_temporal_array_dtype_mismatch`, whose single direct
`DynArray == DynArray` was replaced with the dtype comparison the case is
actually about. That test now builds in 3.2 s and passes. The ~10
nested-equality cases in the 91–120 band remain unbuildable, and the file as a
whole still cannot be run — that is now a known defect with a named cause
rather than an unexplained timeout.
