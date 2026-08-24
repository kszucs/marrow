"""Reproducer: `List` growth silently drops every other `DynScalar`.

Run with:  pixi run -e dev mojo run -I . docs/repros/dynscalar_list_growth.mojo

Expected `int64 int64 int64 int64 int64`; actual `int64 null int64 null int64`.
Elements at odd indices come back as the variant's **first** member
(`NullScalar`), i.e. their discriminant reads as 0. Reserving capacity up front
avoids it, which is what `StructArray.__getitem__` already does
(`arrays.mojo:1930`).

**This is not memory corruption.** Under AddressSanitizer the same run produces
the same wrong values and **no ASAN diagnostic at all** -- no overflow, no
use-after-free. Every access is legal; the moves simply do not all happen. It is
a miscompile of `List`'s reallocation path, not a heap bug. (ASAN also perturbs
it: a second reproduction shape passes under ASAN and fails without it.)

Scope, measured: `List[DynScalar]` and `List[RuntimeValue]` lose elements;
`List[DynArray]` and `List[ArrayData]` do not, though both are equally
recursive. `RuntimeValue` is affected only because its `Payload` variant
contains a `DynScalar`. So recursion is not the trigger -- something specific to
`DynScalar` is, and it is still unidentified. Ruled out by experiment:
the explicit `__deinit__` (`DynArray` has an identical one and is fine),
`IsTriviallyDeinitable` (both report `False`), the `Writable` reflection
defaults, and variant member count (a marrow-free 31-member replica does not
reproduce).
"""

from marrow.scalars import DynScalar, Int64Scalar


def main():
    var xs = List[DynScalar]()  # no capacity -> reallocates while appending
    for i in range(5):
        xs.append(DynScalar(Int64Scalar(Int64(i))))

    var got = String()
    for ref x in xs:
        got += String(x.type())
        got += " "
    print("actual  :", got)
    print("expected: int64 int64 int64 int64 int64")

    var ok = List[DynScalar](capacity=5)  # capacity reserved -> correct
    for i in range(5):
        ok.append(DynScalar(Int64Scalar(Int64(i))))
    var good = String()
    for ref x in ok:
        good += String(x.type())
        good += " "
    print("with capacity:", good)
