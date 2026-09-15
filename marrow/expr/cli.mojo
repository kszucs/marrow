"""`QueryCli` — a plan, run as a command-line program.

Write the plan with `param()` placeholders and hand it over:

```mojo
def query() raises -> DynRelation:
    var orders = scan(param("src", string), schema([field("amount", int64)]))
    return orders.filter(col("amount", int64) >= param("min-amount", int64))


def main() raises:
    QueryCli(query()).run()
```

**The plan is the declaration.** Every parameter the plan reads becomes a
`--name VALUE` option, found by `DynRelation.params()` — the same walk that
answers `columns()` — so the command line cannot drift from the query and
nothing is declared twice. `--help`, `--describe`, `-o/--output`, `--format`
and `--max-rows` are built in.

**Nothing here is paid for unless named.** Each parameter carries a parser
instantiated for its own dtype, so a program with one `int64` parameter links
one parser. `run()` writes text; `run[parquet=True]()` and `run[ipc=True]()`
link the Parquet and Arrow IPC writers, which are the largest thing this layer
can pull in. And no optimizer runs here: a plan prunes because its author wrote
`.optimize[ScanPruning]()` (see CLAUDE.md for what applying it implicitly cost).
"""

from std.sys import argv, exit, stderr

from ..execution import ExecContext
from ..ipc import RecordBatchFileWriter
from ..parquet.writer import write_table
from ..tabular import RecordBatch, Table
from ..utils.argparse import ArgumentParser, ParsedArgs
from .bindings import Bindings, ParamSpec
from .logical import DynRelation


# ---------------------------------------------------------------------------
# Output rendering
# ---------------------------------------------------------------------------


def _repeat(text: String, times: Int) -> String:
    var out = String()
    for _ in range(times):
        out += text
    return out


def _width(text: String) -> Int:
    """Display width in codepoints. `len(String)` is rejected outright in Mojo
    — UTF-8 makes a single length ambiguous — and the byte length would
    mis-align any column holding a non-ASCII value."""
    return len(text.codepoints())


def _pad(var text: String, width: Int) -> String:
    return text + _repeat(String(" "), width - _width(text))


def _cells(batch: RecordBatch, rows: Int) raises -> List[List[String]]:
    """Every rendered cell, row-major, plus a leading header row.

    One pass, because `batch.column(i)[r]` materialises a `DynScalar` and a
    two-pass renderer (measure, then emit) would do it twice. A null renders
    as the empty string here and is spelled by the caller — the table wants
    `null`, CSV wants an empty field, and only one of those can be the
    scalar's own `write_to`.
    """
    var out = List[List[String]]()
    var header = List[String]()
    for ref f in batch.schema.fields:
        header.append(f.name.copy())
    out.append(header^)
    for r in range(rows):
        var row = List[String](capacity=batch.num_columns())
        for c in range(batch.num_columns()):
            ref column = batch.column(c)
            if column.is_null(r):
                row.append(String())
            else:
                row.append(String(column[r]))
        out.append(row^)
    return out^


def render_table(batch: RecordBatch, max_rows: Int = 20) raises -> String:
    """`batch` as an aligned text table, capped at `max_rows` (0 for all).

    This exists because `print(batch)` prints
    `RecordBatch(num_rows=2, schema=...)` and no data at all — fine as a repr,
    useless as the output of a report. A truncated table says so in its footer
    rather than quietly showing a prefix.
    """
    if batch.num_columns() == 0:
        return String("(no columns, ") + String(batch.num_rows()) + " rows)"

    var shown = batch.num_rows()
    if max_rows > 0 and shown > max_rows:
        shown = max_rows
    var grid = _cells(batch, shown)

    var widths = List[Int](capacity=batch.num_columns())
    for c in range(batch.num_columns()):
        widths.append(_width(grid[0][c]))
    for r in range(1, len(grid)):
        for c in range(batch.num_columns()):
            # A null renders as `null`, four columns wide, even though `_cells`
            # handed it back empty.
            var cell = 4 if grid[r][c] == "" else _width(grid[r][c])
            if cell > widths[c]:
                widths[c] = cell

    var out = String()
    for c in range(batch.num_columns()):
        if c > 0:
            out += "  "
        out += _pad(grid[0][c].copy(), widths[c])
    out += "\n"
    for c in range(batch.num_columns()):
        if c > 0:
            out += "  "
        out += _repeat(String("-"), widths[c])
    out += "\n"
    for r in range(1, len(grid)):
        for c in range(batch.num_columns()):
            if c > 0:
                out += "  "
            var cell = String("null") if grid[r][c] == "" else grid[r][c].copy()
            out += _pad(cell^, widths[c])
        out += "\n"

    if shown < batch.num_rows():
        out += (
            String("(")
            + String(shown)
            + " of "
            + String(batch.num_rows())
            + " rows; --max-rows 0 for all)"
        )
    else:
        out += String("(") + String(batch.num_rows()) + " rows)"
    return out^


def _csv_field(var text: String, was_null: Bool) -> String:
    """One CSV field, RFC 4180: quote when the value contains a delimiter, a
    quote or a newline, and double any embedded quote. A null is the empty
    field — the spelling every CSV reader agrees means missing, and the reason
    `_cells` cannot just hand back the scalar's own `null`."""
    if was_null:
        return String()
    elif (
        text.find(",") >= 0
        or text.find('"') >= 0
        or text.find("\n") >= 0
        or text.find("\r") >= 0
    ):
        return String('"') + text.replace('"', '""') + '"'
    else:
        return text^


def render_csv(batch: RecordBatch) raises -> String:
    """`batch` as RFC 4180 CSV with a header row. Never truncated: CSV is what
    a pipe reads, and a silently capped pipe is a data-loss bug."""
    var grid = _cells(batch, batch.num_rows())
    var out = String()
    for r in range(len(grid)):
        for c in range(len(grid[r])):
            if c > 0:
                out += ","
            out += _csv_field(grid[r][c].copy(), r > 0 and grid[r][c] == "")
        out += "\n"
    return out^


# ---------------------------------------------------------------------------
# QueryCli
# ---------------------------------------------------------------------------

comptime _FORMAT_HELP = String(
    "output format: table | csv | parquet | ipc (default: inferred from"
    " --output's extension, else table)"
)


def _program_name() -> String:
    """The name this program was invoked as, exactly as typed — `./orders` is
    how it was run, so it is also how its usage line spells it."""
    var raw = argv()
    if len(raw) == 0:
        return String("query")
    else:
        return String(raw[0])


struct QueryCli(Movable):
    """A plan as a command-line program: every parameter it reads is an option.
    """

    var _plan: DynRelation
    var _params: List[ParamSpec]
    var _parser: ArgumentParser

    def __init__(
        out self, plan: DynRelation, *, description: String = String()
    ) raises:
        """Derive the command line from `plan`'s parameters.

        Raises for a required parameter whose dtype has no command-line
        spelling — that program could never run, so it is refused here rather
        than on every invocation — and, through `ArgumentParser`, for one named
        like a built-in option.

        **Size shaped this surface.** `query_cli` is size-gated at 0.5%, and each
        of these measured more than it was worth: upper-casing the metavar
        (2,752 bytes of `__text`), a separate check for unspellable names
        (1,344), the invoked path's basename (1,200), and validating
        `--max-rows` before the query runs (~960).
        """
        var params = plan.params()
        var parser = ArgumentParser(
            _program_name(), description=description.copy()
        )
        for ref p in params:
            if not p.parse and not p.default:
                raise Error(
                    "QueryCli: parameter '",
                    p.name,
                    "' is ",
                    p.dtype,
                    (
                        ", which cannot be read from the command line; give it"
                        " a default"
                    ),
                )
            parser.option(
                p.name.copy(),
                help=p.help.copy(),
                default=p.default.copy(),
                metavar=String(p.dtype),
                required=not p.default,
            )
        parser.flag(
            String("describe"),
            help=String("print the query plan and exit"),
            short_circuit=True,
        )
        parser.option(
            String("output"),
            short=String("o"),
            metavar=String("PATH"),
            help=String("write results here (default: stdout)"),
            required=False,
        )
        parser.option(
            String("format"),
            metavar=String("FMT"),
            help=_FORMAT_HELP.copy(),
            required=False,
        )
        parser.option(
            String("max-rows"),
            metavar=String("N"),
            help=String("rows to print for `table` output, 0 for all"),
            default=String("20"),
            required=False,
        )
        self._plan = plan.copy()
        self._params = params^
        self._parser = parser^

    def help_text(self) -> String:
        """The `--help` text."""
        return self._parser.help_text()

    def parse(self, tokens: List[String]) raises -> ParsedArgs:
        """`tokens` — argv without the program name — as this program's
        arguments. Raises on any usage error, including a `--format` it cannot
        use, so a misspelled flag is reported before anything runs."""
        var args = self._parser.parse(tokens)
        _ = _resolve_format(args, String())
        return args^

    def bindings(self, args: ParsedArgs) raises -> Bindings:
        """This run's parameter values: each option `args` mentions, parsed as
        its parameter's dtype. An option left out is absent rather than empty,
        so the parameter falls back to its own default."""
        var out = Bindings()
        for ref p in self._params:
            if args.supplied(p.name):
                if not p.parse:
                    raise Error(
                        "--",
                        p.name,
                        ": a ",
                        p.dtype,
                        " parameter cannot be set from the command line",
                    )
                try:
                    out[p.name.copy()] = p.parse.value()(args.get(p.name))
                except e:
                    raise Error("--", p.name, ": ", e)
        return out^

    def run[
        parquet: Bool = False, ipc: Bool = False
    ](self, ctx: ExecContext = ExecContext.auto()) raises:
        """Parse `argv`, then print help, describe the plan, or execute it and
        write the result where `-o` / `--format` say.

        A usage error prints the usage line and **exits 2**; an execution error
        **exits 1**. Both go to stderr, so a failure never lands inside
        `--format csv > out.csv`. `parquet=True` / `ipc=True` link the
        corresponding writer.
        """
        var raw = argv()
        var tokens = List[String](capacity=len(raw))
        for i in range(1, len(raw)):
            tokens.append(String(raw[i]))

        var args: ParsedArgs
        var values = Bindings()
        try:
            args = self.parse(tokens)
            if not args.help_requested and not args.flag(String("describe")):
                values = self.bindings(args)
        except e:
            _exit_with(
                self._parser.usage()
                + "\n"
                + self._parser.prog
                + ": error: "
                + _unprefixed(String(e)),
                2,
            )
            return

        if args.help_requested:
            print(self._parser.help_text())
        elif args.flag(String("describe")):
            print(self._plan)
        else:
            try:
                self._execute[parquet, ipc](args, values, ctx)
            except e:
                _exit_with(
                    self._parser.prog + ": error: " + _unprefixed(String(e)), 1
                )

    def _execute[
        parquet: Bool, ipc: Bool
    ](self, args: ParsedArgs, values: Bindings, ctx: ExecContext) raises:
        """`run` minus the error reporting, so the reporting is one `except`
        rather than one per writer."""
        var path = args.get_or(String("output"), String())
        var fmt = _resolve_format(args, path)
        var batch = self._plan.execute(ctx, values)
        if fmt == "table":
            var rendered = render_table(batch, args.get_int(String("max-rows")))
            if path:
                _write_text(path, rendered^)
            else:
                print(rendered)
        elif fmt == "csv":
            var rendered = render_csv(batch)
            if path:
                _write_text(path, rendered^)
            else:
                print(rendered, end="")
        elif fmt == "parquet":
            comptime if parquet:
                _write_parquet(_require_path(path, fmt), batch)
            else:
                raise Error(
                    "--format parquet: this binary was built without the"
                    " Parquet writer; build it with"
                    " `QueryCli(plan).run[parquet=True]()`"
                )
        else:
            comptime if ipc:
                _write_ipc(_require_path(path, fmt), batch)
            else:
                raise Error(
                    "--format ipc: this binary was built without the Arrow IPC"
                    " writer; build it with `QueryCli(plan).run[ipc=True]()`"
                )


def _resolve_format(args: ParsedArgs, path: String) raises -> String:
    """`--format` if given, else guessed from `path`'s extension, else `table`
    for stdout and `csv` for a file with no telling extension."""
    var explicit = args.get_or(String("format"), String())
    if explicit:
        if (
            explicit == "table"
            or explicit == "csv"
            or explicit == "parquet"
            or explicit == "ipc"
        ):
            return explicit^
        else:
            raise Error(
                "--format: expected table, csv, parquet or ipc, got '",
                explicit,
                "'",
            )
    elif not path:
        return String("table")
    elif path.endswith(".parquet") or path.endswith(".pq"):
        return String("parquet")
    elif path.endswith(".arrow") or path.endswith(".ipc"):
        return String("ipc")
    else:
        return String("csv")


@no_inline
def _exit_with(message: String, status: Int):
    """Report `message` on stderr and end the process with `status` — one
    out-of-line call for every error path, rather than a formatted `print` to
    stderr inlined into each."""
    print(message, file=stderr)
    exit(status)


def _unprefixed(var message: String) -> String:
    """`message` without a leading `argparse: `.

    `ArgumentParser` names itself so its errors read well when a caller prints
    them raw. Here the program has already named itself — `orders: error: ...`
    — and `orders: error: argparse: unrecognized option` names a module the
    user has never heard of."""
    if message.startswith("argparse: "):
        return String(message.removeprefix("argparse: "))
    else:
        return message^


def _require_path(path: String, fmt: String) raises -> String:
    if path:
        return path.copy()
    else:
        raise Error(
            "--format ",
            fmt,
            (
                " needs an output file: pass -o PATH (a binary format cannot go"
                " to stdout)"
            ),
        )


def _write_text(path: String, var text: String) raises:
    with open(path, "w") as f:
        f.write(text^)


def _write_parquet(path: String, batch: RecordBatch) raises:
    """Kept out of `run` so the only call to it sits inside a `comptime if`
    branch: when that branch is deleted this function becomes unreferenced and
    the Parquet writer goes with it."""
    write_table(Table.from_batches(batch.schema, [batch.copy()]), path)


def _write_ipc(path: String, batch: RecordBatch) raises:
    var writer = RecordBatchFileWriter(path, batch.schema)
    writer.write_batch(batch)
    writer.close()
