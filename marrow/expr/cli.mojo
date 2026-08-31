"""`QueryCli` — the AOT lane's product surface: a plan, as a command-line program.

The comptime lane compiles an expression tree into one fused SIMD loop with no
interpreter left in the binary. This module is what makes that *usable*: it
turns a plan into a program with `--help`, late-bound parameters, `--describe`
and an output writer, without giving any of that back in binary size.

## The shape

Declare, parse, build, run — in that order, in one `main()`:

```mojo
def main() raises:
    var cli = QueryCli("sales", description="Orders above a threshold.")
    var min_amount = cli.param("min-amount", int64, default=0, help="lower bound")
    cli.argument("src", help="input Parquet file")

    if cli.parse():
        var sch = schema([field("id", int64), field("amount", int64)])
        var plan = (
            scan(cli.get("src"), sch^)
            .filter(col("amount", int64) > min_amount)
            .optimize[AllRules]()
        )
        cli.run(plan^)
```

`cli.param(...)` is the whole trick. It returns an ordinary `Param[T]` — the
same node `param()` builds, usable anywhere a `Literal[T]` is — *and* records
the declaration, so one line produces the plan node, the `--min-amount` option,
its `--help` entry and the string-to-scalar coercion that binds it. The old
layer made the author write the declaration twice, once for the parser and once
for the plan, and nothing checked that the two agreed.

## Why the declarations live here and not on the plan

`leaves.mojo` argues at length that a logical node is stateless and that a plan
therefore cannot be asked what parameters it takes — there is no `params()`
traversal, and a parameter's *value* travels through `Bindings` rather than in a
cell. Both still hold. `QueryCli` is not a registry bolted onto the plan: it is
the *declaration site*, and it is an ordinary local value, so there is no
process-global to leak into the next plan's `--help` and no global to race on.
The plan stays immutable and stays ignorant of the CLI.

## What it costs

Nothing a program does not name.

- **Coercion is monomorphic.** `cli.param("min-a", int64)` instantiates one
  `_coerce_param[Int64Type]` and stores it as a thin function pointer. A
  program declaring one `int64` parameter links one coercion, not a
  `dispatch_numeric` ladder over all eleven widths — the same closed-erasure
  property that keeps `kernels::sort` out of a binary that never sorts.
- **The file writers are comptime-gated.** `run()` writes text; `run[parquet=True]()`
  and `run[ipc=True]()` opt into the Parquet and Arrow IPC writers, which are
  large. The gate is a comptime parameter at the call site rather than a
  `-D` define, so the source says which formats the binary supports and
  `comptime if` deletes the rest.
- **Nothing here reaches the runtime lane.** `QueryCli` names `DynRelation`,
  `Bindings` and `DynScalar`; it never constructs a `RuntimeValue`, so a fused
  program stays fused.

## The built-in arguments

Every `QueryCli` gets `-h/--help`, `--describe`, `-o/--output`, `--format` and
`--max-rows` without the author declaring them. `--describe` prints the plan and
exits — worth its near-zero cost because a compiled binary is otherwise opaque
about the query baked into it, and it short-circuits the required-argument
check, so `./q --describe` works with no arguments at all.
"""

from std.sys import argv, exit

from ..dtypes import NumericType
from ..scalars import DynScalar, PrimitiveScalar
from ..schema import Schema
from ..tabular import RecordBatch, Table
from ..utils.argparse import ArgumentParser, ParsedArgs
from ..execution import ExecContext
from ..ipc import RecordBatchFileWriter
from ..parquet.writer import write_table
from .bindings import Bindings
from .logical import DynRelation
from .`comptime`.leaves import Param


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
            var cell = String("null") if grid[r][c] == "" else grid[r][
                c
            ].copy()
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
# Parameter coercion
# ---------------------------------------------------------------------------


def _coerce_param[
    T: NumericType
](name: String, text: String) raises -> DynScalar:
    """One command-line token as a `T` scalar, naming the option when it is not
    one.

    Instantiated once per dtype a program actually declares — this is the whole
    reason the coercion is a monomorphic function stored as a pointer rather
    than a runtime `dispatch_numeric`, which would link every width.
    """
    comptime if T.native.is_floating_point():
        try:
            return PrimitiveScalar[T](Scalar[T.native](atof(text))).to_dyn()
        except:
            raise Error(
                "--", name, " expects a number, got '", text, "'"
            )
    else:
        try:
            return PrimitiveScalar[T](Scalar[T.native](atol(text))).to_dyn()
        except:
            raise Error(
                "--", name, " expects an integer, got '", text, "'"
            )


struct _ParamDecl(Copyable, Movable):
    """A declared parameter: its name, and how to turn a token into its scalar.

    The dtype is *not* stored — it is baked into `coerce`, which is what keeps
    binding closed over the declared dtypes instead of open over all of them.
    """

    var name: String
    var coerce: def (String, String) thin raises -> DynScalar

    def __init__(
        out self,
        var name: String,
        coerce: def (String, String) thin raises -> DynScalar,
    ):
        self.name = name^
        self.coerce = coerce


# ---------------------------------------------------------------------------
# QueryCli
# ---------------------------------------------------------------------------

comptime _FORMAT_HELP = String(
    "output format: table | csv | parquet | ipc (default: inferred from"
    " --output's extension, else table)"
)


struct QueryCli(Movable):
    """A compiled plan's command-line surface: declare, `parse`, `run`."""

    var _parser: ArgumentParser
    var _params: List[_ParamDecl]
    var _args: ParsedArgs
    var _parsed: Bool

    def __init__(
        out self, var prog: String, *, var description: String = String()
    ) raises:
        self._parser = ArgumentParser(prog^, description=description^)
        self._params = List[_ParamDecl]()
        self._args = ParsedArgs(
            values={}, flags={}, supplied={}, trailing=[], help_requested=False
        )
        self._parsed = False
        self._parser.flag(
            String("describe"),
            help=String("print the query plan and exit"),
            short_circuit=True,
        )
        self._parser.option(
            String("output"),
            short=String("o"),
            metavar=String("PATH"),
            help=String("write results here (default: stdout)"),
            required=False,
        )
        self._parser.option(
            String("format"),
            metavar=String("FMT"),
            help=_FORMAT_HELP.copy(),
            required=False,
        )
        self._parser.option(
            String("max-rows"),
            metavar=String("N"),
            help=String("rows to print for `table` output, 0 for all"),
            default=String("20"),
            required=False,
        )

    # -- declaration --------------------------------------------------------

    def param[
        T: NumericType
    ](
        mut self,
        var name: String,
        dtype: T,
        *,
        var default: Optional[Scalar[T.native]] = None,
        var help: String = String(),
        var metavar: String = String(),
    ) raises -> Param[T]:
        """Declare `--name VALUE` **and** return the plan node it binds.

        Required unless it has a `default`. The default lives on the returned
        node — the parser is told about it only so `--help` can show it, and
        binding skips any option argv did not mention, so there is exactly one
        source of truth for the value.
        """
        var shown = Optional[String](None)
        if default:
            shown = String(default.value())
        self._parser.option(
            name.copy(),
            help=help.copy(),
            default=shown^,
            metavar=metavar^ if metavar else String("N"),
            required=not default,
        )
        self._params.append(_ParamDecl(name.copy(), _coerce_param[T]))
        return Param[T](name^, help^, default^)

    def argument(
        mut self, var name: String, *, var help: String = String()
    ) raises:
        """Declare a positional argument — a path, typically. Read it back with
        `get(name)` after `parse()`."""
        self._parser.positional(name^, help=help^)

    def option(
        mut self,
        var name: String,
        *,
        var short: String = String(),
        var default: Optional[String] = None,
        var help: String = String(),
        var metavar: String = String(),
    ) raises:
        """Declare `--name VALUE` as a plain string, for a value the plan does
        not read as a scalar (an output prefix, a partition name). Read it back
        with `get(name)`."""
        self._parser.option(
            name^,
            short=short^,
            help=help^,
            default=default^,
            metavar=metavar^,
            required=not default,
        )

    def flag(
        mut self,
        var name: String,
        *,
        var short: String = String(),
        var help: String = String(),
    ) raises:
        """Declare `--name` as a valueless boolean. Read it back with
        `flag(name)`."""
        self._parser.flag(name^, short=short^, help=help^)

    # -- lifecycle ----------------------------------------------------------

    def parse(mut self) raises -> Bool:
        """Parse `argv`. True to go on and build the plan, False when `--help`
        was handled and the program should stop.

        A usage error is reported here and **exits 2** rather than propagating:
        a shipped binary that answers a misspelled flag with
        `Unhandled exception caught during execution` has failed its user, and
        every compiled query would otherwise repeat the same handling. That is
        also why `--help` prints here instead of being handed back as a flag —
        a program that forgot to check it would answer `--help` by running the
        query. `ArgumentParser` itself stays pure and testable; this is the
        layer where exiting is the right answer.
        """
        var raw = argv()
        var tail = List[String](capacity=len(raw))
        for i in range(1, len(raw)):
            tail.append(String(raw[i]))
        try:
            self._args = self._parser.parse(tail)
            self._parsed = True
            # Validated here rather than where it is read: a misspelled
            # `--format` is a usage error, and reporting it from `run()` would
            # give it an execution error's exit code after the query had
            # already run.
            _ = self._resolve_format(String())
        except e:
            print(self._parser.usage())
            print(self._parser.prog + ": error: " + _unprefixed(String(e)))
            exit(2)
        if self._args.help_requested:
            print(self._parser.help_text())
            return False
        else:
            return True

    def _require_parsed(self) raises:
        if not self._parsed:
            raise Error(
                "QueryCli: call parse() before reading arguments or running a"
                " plan"
            )

    def _describing(self) -> Bool:
        """Whether `--describe` short-circuited this run."""
        try:
            return self._args.flag(String("describe"))
        except:
            return False

    def get(self, name: String) raises -> String:
        """A declared positional's or string option's value.

        Answers the empty string under `--describe`. A short-circuiting flag
        suppresses the required-argument check, which is the whole point of it
        — `./q --describe` has to work with no arguments — so the plan is still
        built, just over an empty path it never opens.
        """
        self._require_parsed()
        if self._describing():
            return self._args.get_or(name, String())
        else:
            return self._args.get(name)

    def get_or(self, name: String, var fallback: String) raises -> String:
        """`get(name)`, or `fallback` when it has no value."""
        self._require_parsed()
        return self._args.get_or(name, fallback^)

    def flag(self, name: String) raises -> Bool:
        """Whether a declared flag appeared."""
        self._require_parsed()
        return self._args.flag(name)

    def bindings(self) raises -> Bindings:
        """This run's parameter values, for a caller that wants to execute the
        plan itself rather than through `run()`.

        Only options argv actually mentioned appear. An omitted one is not an
        empty binding — it is absent, so `Param.bind` falls back to the node's
        own default or raises naming itself.
        """
        self._require_parsed()
        var out = Bindings()
        for ref decl in self._params:
            if self._args.supplied(decl.name):
                out[decl.name.copy()] = decl.coerce(
                    decl.name, self._args.get(decl.name)
                )
        return out^

    def _resolve_format(self, path: String) raises -> String:
        """`--format` if given, else guessed from `path`'s extension, else
        `table` for stdout and `csv` for a file with no telling extension."""
        var explicit = self._args.get_or(String("format"), String())
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

    def run[
        parquet: Bool = False, ipc: Bool = False
    ](
        mut self,
        var plan: DynRelation,
        ctx: ExecContext = ExecContext.auto(),
    ) raises:
        """Handle `--describe`, execute `plan` with this run's bindings, and
        write the result where `-o` / `--format` say.

        `parquet=True` / `ipc=True` link the corresponding writer. They are off
        by default because they are the largest thing this layer can pull into
        a binary, and a query that prints or pipes its result should not pay
        for them — a `--format parquet` against a build that did not ask for it
        raises naming the flag that turns it on.
        """
        self._require_parsed()
        if self._describing():
            print(plan)
        else:
            try:
                self._execute[parquet, ipc](plan^, ctx)
            except e:
                print(self._parser.prog + ": error: " + _unprefixed(String(e)))
                exit(1)

    def _execute[
        parquet: Bool, ipc: Bool
    ](mut self, var plan: DynRelation, ctx: ExecContext) raises:
        """`run` minus the error reporting, so the reporting is one `except`
        rather than one per writer."""
        var path = self._args.get_or(String("output"), String())
        var fmt = self._resolve_format(path)
        var batch = plan.execute(ctx, self.bindings())
        if fmt == "table":
            var rendered = render_table(
                batch, self._args.get_int(String("max-rows"))
            )
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
                    " Parquet writer; build it with `cli.run[parquet=True]"
                    "(plan)`"
                )
        else:
            comptime if ipc:
                _write_ipc(_require_path(path, fmt), batch)
            else:
                raise Error(
                    "--format ipc: this binary was built without the Arrow"
                    " IPC writer; build it with `cli.run[ipc=True](plan)`"
                )

    def help_text(self) -> String:
        """The `--help` text, for a caller that renders it somewhere else."""
        return self._parser.help_text()


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
            " needs an output file: pass -o PATH (a binary format cannot go to"
            " stdout)",
        )


def _write_text(path: String, var text: String) raises:
    with open(path, "w") as f:
        f.write(text^)


def _write_parquet(path: String, batch: RecordBatch) raises:
    """Kept out of `run` so the only call to it sits inside a `comptime if`
    branch: when that branch is deleted this function becomes unreferenced and
    the Parquet writer goes with it. An inline call would be gated just as
    well, but this keeps the gate visible in one line."""
    write_table(Table.from_batches(batch.schema, [batch.copy()]), path)


def _write_ipc(path: String, batch: RecordBatch) raises:
    var writer = RecordBatchFileWriter(path, batch.schema)
    writer.write_batch(batch)
    writer.close()
