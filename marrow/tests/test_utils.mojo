from std.testing import assert_false, assert_true
from marrow.testing import TestSuite
from marrow.utils import has_accelerator_support
from std.sys import CompilationTarget
from std.sys import has_accelerator


def test_has_accelerator_support() raises:
    # float64 is supported on CUDA (Linux) but not on Metal (Apple Silicon).
    if has_accelerator() and not CompilationTarget.is_apple_silicon():
        assert_true(has_accelerator_support[DType.float64]())
    else:
        assert_false(has_accelerator_support[DType.float64]())


def main() raises:
    TestSuite.run[__functions_in_module()]()
