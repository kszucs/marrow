"""Debug - test BitmapView(Bitmap) constructor specifically."""

from marrow.buffers import Bitmap, Buffer
from marrow.views import BitmapView


def main() raises:
    var b = Bitmap.alloc_zeroed(16)
    b.set_range(0, 16, True)
    var bm = b.to_immutable(16)

    print("Before any BitmapView:")
    print("  byte0:", bm._buffer.ptr[0])

    # Create BitmapView from Bitmap, let it die
    print("\n--- Creating BitmapView(bm) #1 ---")
    var v1 = BitmapView(bm)
    print("  view byte0:", v1._data[0])
    _ = v1^  # destroy view

    # Stress allocator
    for _ in range(100):
        var tmp = Buffer.alloc_zeroed(64)
        _ = tmp^

    print("After view #1 destroyed + stress:")
    print("  byte0:", bm._buffer.ptr[0])

    # Create BitmapView as temporary
    print("\n--- BitmapView(bm).test(0) ---")
    var result = BitmapView(bm).test(0)
    print("  test(0):", result)

    # Stress allocator
    for _ in range(100):
        var tmp = Buffer.alloc_zeroed(64)
        _ = tmp^

    print("After BitmapView temporary + stress:")
    print("  byte0:", bm._buffer.ptr[0])

    # Now count_set_bits
    print("\n--- BitmapView(bm).count_set_bits() ---")
    var count = BitmapView(bm).count_set_bits()
    print("  count:", count, "(expected 16)")
