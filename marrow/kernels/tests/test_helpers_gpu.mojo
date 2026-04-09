from std.testing import assert_false, assert_true
from marrow.testing import TestSuite
from marrow.kernels.helpers import has_accelerator_support
from std.sys import CompilationTarget


def test_has_accelerator_support() raises:
    assert_true(has_accelerator_support[DType.float32]())
    if CompilationTarget.is_apple_silicon():
        assert_false(has_accelerator_support[DType.float64]())
    else:
        assert_true(has_accelerator_support[DType.float64]())


def main() raises:
    TestSuite.run[__functions_in_module()]()
