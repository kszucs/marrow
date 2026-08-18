"""Does ASAN intercept Mojo heap allocations? Deliberate overflow."""

from std.memory.alloc import unsafe_alloc


def main():
    var p = unsafe_alloc[UInt8](16, alignment=64)
    for i in range(200):  # far past the end
        p[i] = UInt8(i)
    print("wrote", p[199])
    p.unsafe_free()
