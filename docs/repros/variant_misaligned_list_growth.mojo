"""`Variant`'s `size_of` disagrees with its real allocation stride when the
largest member is not the most-aligned member.

Run:  mojo run variant_misaligned_list_growth.mojo   (pure stdlib, no marrow)
Observed on mojo 1.1.0.dev2026082305, macOS arm64 (Darwin 24.6.0).

`Variant[NoneType, String, Wide, Aligned]` reports `size_of == 96` and
`align_of == 32`, but `List` lays its elements out **112 bytes** apart. 112 is
not even a multiple of the type's own 32-byte alignment. Note:

    96  == round_up(72 + 1, 32)     <- what `size_of` reports
    112 == round_up(96 + 1, 16)     <- the stride actually used

i.e. two layout computations disagree about the alignment. Control types in the
same program (`Int`, `String`, `Wide`, and an over-aligned `Aligned` that *is*
its own largest member) all have stride == size_of.

**Located in the compiler.** `!kgen.variant` lowers to a struct of
`{union, discriminant}` (`KGEN/lib/Transforms/LowerCallingConventions.cpp`,
`lowerVariantType`), and a union's size is "max member size, rounded up to the
union's alignment". But `VariantType::getTypeSize`
(`KGEN/lib/KGENDialect/KGENTypes.cpp`) computes:

    alignTo(alignTo(contentSize, discrAlign) + discrSize, align)

It rounds the content to **`discrAlign`** -- the *discriminant's* alignment,
which is 1 for an `i8` -- rather than to the union's alignment. So:

    size_of : alignTo(alignTo(72, 1)  + 1, 32) = alignTo(73, 32) = 96
    reality : union = alignTo(72, 32) = 96, discr at 96      -> 112

When the largest member *is* the most-aligned one, `contentSize` is already a
multiple of the union's alignment and the two agree -- which is exactly why the
bug needs "largest != most-aligned". The comment directly above the faulty line
reasons about the discriminant's *size* but then uses its *alignment*, and
`getContentSize` carries its own `FIXME` about misusing `getTypeAllocSize`.

The consequence: growing a `List` of such a variant silently drops elements.
Both the move loop and the reads use pointer arithmetic, and where the two
strides disagree, elements land at the wrong slots -- appending "a".."e" to an
unreserved `List` reads back `a_c_e`, positions 1 and 3 never written.
Reserving capacity up front avoids the reallocation and hides it.

**Not a memory-safety bug**: under AddressSanitizer the values are equally
wrong and there is *no* diagnostic. ASAN also perturbs it -- some shapes pass
under ASAN and fail without it -- so do not use ASAN to confirm this class.

The trigger needs BOTH properties; neither alone reproduces:

  | variant shape                                   | size/align | stride |
  |-------------------------------------------------|-----------|--------|
  | largest member is also the most-aligned          |  96 / 32  |  96 ok |
  | every member 8-aligned                           |  80 / 8   |  80 ok |
  | largest 8-aligned, a smaller member 32-aligned   |  96 / 32  | 112 BAD|

Rows 1 and 3 have identical size *and* alignment, so neither explains it alone.

In marrow this is `DynScalar`: its largest member `StructScalar` is 72 bytes at
align 8, while `Decimal128Scalar` (int128) is 48 bytes at align 16 and
`Decimal256Scalar` (int256) is 64 bytes at align 32. Any decimal scalar with
alignment above 8 puts `DynScalar` into the broken shape.
"""

from std.sys.info import align_of, size_of
from std.utils import Variant


struct Wide(Copyable, Movable):
    """72 bytes, align 8 -- the largest member, but weakly aligned."""

    var a: Int
    var b: Int
    var c: Int
    var d: Int
    var e: Int
    var f: Int
    var g: Int
    var h: Int
    var i: Int

    def __init__(out self):
        self.a = 0
        self.b = 0
        self.c = 0
        self.d = 0
        self.e = 0
        self.f = 0
        self.g = 0
        self.h = 0
        self.i = 0


struct Aligned(Copyable, Movable):
    """64 bytes, align 32 -- smaller, but more strictly aligned."""

    var v: SIMD[DType.int256, 1]
    var valid: Bool

    def __init__(out self):
        self.v = SIMD[DType.int256, 1](0)
        self.valid = True


comptime Mixed = Variant[NoneType, String, Wide, Aligned]


def _stride[T: Copyable & Deinitable](var a: T, var b: T) -> Int:
    var xs = List[T](capacity=2)
    xs.append(a^)
    xs.append(b^)
    var d = Int(Pointer(to=xs[1])) - Int(Pointer(to=xs[0]))
    _ = xs^
    return d


def main():
    print("type      size  align  stride")
    print("Int      ", size_of[Int](), "   ", align_of[Int](), "   ", _stride(1, 2))
    print("String   ", size_of[String](), "  ", align_of[String](), "   ",
          _stride(String("a"), String("b")))
    print("Wide     ", size_of[Wide](), "  ", align_of[Wide](), "   ",
          _stride(Wide(), Wide()))
    print("Aligned  ", size_of[Aligned](), "  ", align_of[Aligned](), "  ",
          _stride(Aligned(), Aligned()))
    print("Mixed    ", size_of[Mixed](), "  ", align_of[Mixed](), "  ",
          _stride(Mixed(String("a")), Mixed(String("b"))),
          "  <-- stride != size_of")
    print()

    var xs = List[Mixed]()  # no capacity -> reallocates
    for ref n in [String("a"), String("b"), String("c"), String("d"), String("e")]:
        xs.append(Mixed(n.copy()))
    var got = String()
    for ref x in xs:
        got += x[String] if x.isa[String]() else String("_")
    print("append a..e to an unreserved List:", got)
    print("expected                         : abcde")
