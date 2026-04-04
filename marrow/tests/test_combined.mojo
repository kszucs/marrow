from std.testing import TestSuite
from marrow.tests.test_dtypes import test_cases as dtypes_cases
from marrow.tests.test_views import test_cases as views_cases

def main() raises:
    TestSuite.discover_tests[dtypes_cases.concat(views_cases)]().run()
