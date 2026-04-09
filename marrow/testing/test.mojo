"""Test suite wrapper with ``--list`` and ``--json`` support.

Wraps ``std.testing.TestSuite`` to add:
- ``--list``  Print JSON array of test names and exit.
- ``--json``  Print JSON test results (for pytest consumption).

Usage in test files::

    from std.testing import assert_equal, assert_true
    from marrow.testing import TestSuite

    def test_something() raises:
        assert_equal(1 + 1, 2)

    def main() raises:
        TestSuite.run[__functions_in_module()]()
"""

from std.testing import TestSuite as _StdTestSuite
from std.testing.suite import TestResult, TestSuiteReport
from std.reflection import get_function_name
from std.sys import argv
from std.sys.intrinsics import _type_is_eq


struct CLIFlags:
    """Parsed CLI flags for test suites."""

    var list_cases: Bool
    var json_output: Bool
    var args: List[StaticString]

    def __init__(out self):
        self.args = List[StaticString](argv())
        self.list_cases = False
        self.json_output = False

        var i = 1
        while i < len(self.args):
            if self.args[i] == "--list":
                self.list_cases = True
            elif self.args[i] == "--json":
                self.json_output = True
            i += 1


def _print_json_array(names: List[String]):
    """Print a JSON array of strings to stdout."""
    print("[")
    for i in range(len(names)):
        var comma = "," if i < len(names) - 1 else ""
        print('  "' + names[i] + '"' + comma)
    print("]")


struct TestSuite:
    """Drop-in replacement for ``std.testing.TestSuite``.

    Adds ``--list`` (print case names as JSON) and ``--json`` (print
    results as JSON) on top of the standard suite.
    """

    @staticmethod
    def _collect_names[funcs: Tuple, prefix: StaticString]() -> List[String]:
        """Collect function names matching *prefix* at compile time."""
        var names = List[String]()
        comptime for idx in range(len(funcs)):
            comptime func = funcs[idx]
            comptime if get_function_name[func]().startswith(prefix):
                names.append(get_function_name[func]())
        return names^

    @staticmethod
    def _strip_flags(flags: CLIFlags) -> List[StaticString]:
        """Build cli_args with our flags stripped for std TestSuite."""
        var cli_args = List[StaticString]()
        for arg in flags.args:
            if arg != "--list" and arg != "--json":
                cli_args.append(arg)
        return cli_args^

    @staticmethod
    def run[funcs: Tuple, /]() raises:
        """Discover tests, optionally list them, and run.

        Parameters:
            funcs: Pass ``__functions_in_module__()``.
        """
        var flags = CLIFlags()

        if flags.list_cases:
            _print_json_array(Self._collect_names[funcs, "test_"]())
            return

        if flags.json_output:
            var suite = _StdTestSuite.discover_tests[funcs](
                cli_args=Self._strip_flags(flags)
            )
            var report: TestSuiteReport
            try:
                report = suite.generate_report()
            finally:
                suite^.abandon()

            # Emit JSON results.
            print("[")
            for i in range(len(report.reports)):
                ref r = report.reports[i]
                var status: String
                if r.result == TestResult.PASS:
                    status = "PASS"
                elif r.result == TestResult.FAIL:
                    status = "FAIL"
                else:
                    status = "SKIP"
                var error_str = String("")
                if r.error:
                    var raw = String(r.error.value())
                    error_str = raw.replace("\\", "\\\\").replace(
                        '"', '\\"'
                    ).replace("\n", "\\n")
                var comma = "," if i < len(report.reports) - 1 else ""
                print(
                    '  {"name": "' + r.name
                    + '", "status": "' + status
                    + '", "duration_ns": ' + String(r.duration_ns)
                    + ', "error": "' + error_str
                    + '"}' + comma
                )
            print("]")

            if report.failures > 0:
                raise Error(report^)
            return

        # Default: delegate to std TestSuite (human-readable output).
        _StdTestSuite.discover_tests[funcs](
            cli_args=Self._strip_flags(flags)
        ).run()
