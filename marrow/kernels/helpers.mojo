from std.sys import has_accelerator, CompilationTarget


def has_accelerator_support[dtype: DType]() -> Bool:
    """Check if there is accelerator support for the given dtype.

    For example Metal doesn't support float64 as of April 2026.
    """
    if not has_accelerator():
        return False
    if not CompilationTarget.is_apple_silicon():
        return True
    return dtype != DType.float64
