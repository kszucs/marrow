"""Command-line argument parsing: `argv` in, named values out, `--help` for free.

A **leaf module** — it imports nothing, from marrow or from `std` — which is
the property every other file under `utils/` has and the reason this one is
usable from a Parquet CLI, a benchmark driver or a compiled query program
without any of them depending on each other.

This was extracted from the previous expression layer's parameter module,
where the generic half of argv handling (`parse_params`, `render_usage`,
`split_cli_args`) had grown into its late-bound-parameter registry. That layer
has since been deleted; this half survived because it never needed it. The split is drawn at
**what a token means**:

- *Here*: which tokens are options, which are flags, which are positionals;
  `--name value` versus `--name=value`; short aliases; the `--` terminator;
  defaults; required-ness; the unknown-option and missing-value errors; and the
  `--help` text describing all of it. Values are answered as `String`, with
  `get_int` / `get_float` / `get_bool` for the three conversions every CLI
  needs.
- *Not here*: anything that has to know a `DataType` to interpret a token.
  the old `_parse_scalar` turned `"1560601845"` into a
  `PrimitiveScalar[TimestampType]` by dispatching on the declared parameter's
  dtype — that is expression-layer knowledge, it needs `marrow.dtypes` and
  `marrow.scalars`, and importing either would cost the leaf property. It
  belongs on top of this: build a parser from the declarations, run
  it, then coerce each answered string.
- *Also not here*: `--describe`'s JSON, and the `-o` / `--format` output
  writers. The JSON payload's whole content is the **dtype** of each parameter,
  so a generic `describe()` could not emit it, and the JSON-escaping helper it
  needs is a string utility rather than argument parsing — putting it here
  would make this module two things. The writers know about Parquet and IPC.
  Both stay in `params.mojo`.

`split_cli_args` stays there too, for a different reason: it is generic
mechanically (pull two flags out of a token list) but its content is the
*policy* that `-o` names an output path and `--format` overrides the writer.
`ArgumentParser` subsumes the mechanism —
`p.option("output", short="o"); p.option("format", default="")` is the whole
of it — so whoever re-adds `execute_cli` should delete it rather than port it.

## Shape

Register, then run:

```mojo
var p = ArgumentParser("query", description="Run a compiled plan.")
p.option("min-a", metavar="N", help="lower bound")
p.option("limit", metavar="N", default="10", help="row cap")
p.flag("verbose", short="v", help="chatter")
p.positional("src", help="input path")

var args = p.parse(argv_tail)
if args.help_requested:
    print(p.help_text())
else:
    run(args.get("src"), args.get_int("min-a"), args.flag("verbose"))
```

Deliberate departures from Python's `argparse`, each of which cost something to
decide:

- **An option with no `default` is required.** Python's `argparse` defaults
  options to optional and makes `required=True` the opt-in; a query binary's
  parameters are the other way round — the common case is a value the program
  cannot run without. `required=False` opts out, and passing a `default` opts
  out implicitly (an argument that can always answer is never missing).
- **Unknown options and surplus positionals are errors, never ignored.** A
  typo that silently parses as "no value supplied" is the failure mode this
  whole module exists to prevent.
- **`-h` / `--help` are built in but do not print or exit.** A library that
  writes to stdout and calls `exit()` cannot be tested without a subprocess, and
  the code it replaces was factored into `List[String]` in / value out for
  exactly that reason. `parse` sets `ParsedArgs.help_requested` and skips
  the required-argument check; the caller decides what to do. Declaring an
  argument named `help` (or short `h`) shadows the built-in.
- **Any flag can be marked `short_circuit`.** `--help` is not the only flag
  that means "do not run the program": the old `--describe` is another, and
  both must be usable *without* supplying the required arguments they are
  asking about. A short-circuiting flag suppresses the required check exactly
  as `--help` does. Defaults are still applied — they cost nothing and a
  `--describe` implementation wants to see them.
- **No short-option bundling** (`-abc` for `-a -b -c`). It is ambiguous against
  the `-o value` form the tree already uses, and nothing here needs it.
  `-o value` and `-o=value` both work; `-ovalue` is an unrecognized option.

A token that starts with `-` and is longer than one character is an option, so
a negative number cannot be a positional; `--` is the escape hatch, and a lone
`-` is an ordinary positional (the conventional stdin spelling). Everything
after `--` fills the remaining positionals and then the trailing argument, if
one is declared.

Lookup is a linear scan over the declarations rather than a `Dict`. A parser
has a handful of arguments and is run once per process, so the scan is not
measurable, and it keeps declaration order — which is what `usage()` and
`help_text()` render, and what positionals are matched in.
"""


# ---------------------------------------------------------------------------
# Token conversions
# ---------------------------------------------------------------------------


def parse_bool(text: String) raises -> Bool:
    """`true`/`false`/`1`/`0`, case-insensitively.

    The exact spelling set the old `_parse_scalar` accepted for a `bool`
    parameter, kept identical so a CLI flag reads the same whichever layer
    interprets it. `yes`/`no`/`on`/`off` are deliberately absent: every extra
    spelling is one more thing two implementations can disagree about, and a
    flag (`p.flag(...)`) is the right way to spell a boolean that has no value.
    """
    var lower = text.lower()
    if lower == "true" or lower == "1":
        return True
    elif lower == "false" or lower == "0":
        return False
    else:
        raise Error(
            "expected a bool ('true'/'false'/'1'/'0'), got '" + text + "'"
        )


# ---------------------------------------------------------------------------
# ArgSpec
# ---------------------------------------------------------------------------


struct ArgSpec(Copyable, Movable):
    """One declared argument: what it is called, what kind it is, and how it
    renders in `--help`.

    Constructed through `ArgumentParser.option` / `.flag` / `.positional` /
    `.trailing` rather than directly — the four constrain which fields are
    meaningful (a flag has no default, a trailing argument is never required),
    and a single public constructor taking all of them would let those
    combinations be spelled.
    """

    comptime OPTION: UInt8 = 0
    """`--name value`, `--name=value`, or `-s value` for a declared short."""

    comptime FLAG: UInt8 = 1
    """`--name`, present or absent, never carrying a value."""

    comptime POSITIONAL: UInt8 = 2
    """Matched by position among the non-option tokens."""

    comptime TRAILING: UInt8 = 3
    """Every positional token past the last declared `POSITIONAL`."""

    var name: String
    var short: String
    var kind: UInt8
    var help: String
    var default: Optional[String]
    var metavar: String
    var required: Bool
    var short_circuit: Bool

    def __init__(
        out self,
        *,
        var name: String,
        kind: UInt8,
        var short: String = String(),
        var help: String = String(),
        var default: Optional[String] = None,
        var metavar: String = String(),
        required: Bool = True,
        short_circuit: Bool = False,
    ):
        self.name = name^
        self.short = short^
        self.kind = kind
        self.help = help^
        self.default = default^
        self.metavar = metavar^
        self.required = required
        self.short_circuit = short_circuit

    def is_required(self) -> Bool:
        """Whether `parse` must see this argument.

        A default makes it optional whatever `required` says: an argument that
        can always answer is never missing. Flags and the trailing argument are
        never required — absence is a meaningful answer for both.
        """
        if self.kind == Self.FLAG or self.kind == Self.TRAILING:
            return False
        else:
            return self.required and not self.default

    def is_option_like(self) -> Bool:
        """Whether this argument is spelled with a leading dash."""
        return self.kind == Self.OPTION or self.kind == Self.FLAG

    def spelling(self) -> String:
        """How to name this argument back to the user in an error message."""
        if self.is_option_like():
            return "--" + self.name
        else:
            return self.name.copy()

    def placeholder(self) -> String:
        """The value stand-in in usage text — `metavar` when declared, else
        `VALUE`, which says "this takes one argument" without pretending to
        know its type. Callers that do know pass one:
        `metavar=String(dtype)`."""
        if self.metavar.byte_length() > 0:
            return self.metavar.copy()
        else:
            return String("VALUE")


# ---------------------------------------------------------------------------
# ParsedArgs
# ---------------------------------------------------------------------------


struct ParsedArgs(Copyable, Movable):
    """The result of `ArgumentParser.parse`: resolved values, flag states, and
    whatever `--` handed through.

    Values and flags are kept in **two** tables rather than one keyed by name,
    so that `get("verbose")` on a flag and `flag("limit")` on an option are
    both named errors instead of a string that happens to read `"true"`. A
    third table records which names argv actually mentioned, which is the only
    way to tell "took the default" from "was given the default's value".
    """

    var _values: Dict[String, String]
    var _flags: Dict[String, Bool]
    var _supplied: Dict[String, Bool]
    var _trailing: List[String]
    var help_requested: Bool
    """Set when the built-in `-h` / `--help` appeared. `parse` skips the
    required-argument check when it does, so the caller can print
    `ArgumentParser.help_text()` and stop."""

    def __init__(
        out self,
        var values: Dict[String, String],
        var flags: Dict[String, Bool],
        var supplied: Dict[String, Bool],
        var trailing: List[String],
        help_requested: Bool,
    ):
        self._values = values^
        self._flags = flags^
        self._supplied = supplied^
        self._trailing = trailing^
        self.help_requested = help_requested

    def get(self, name: String) raises -> String:
        """The value of an option or positional, raising if it has none.

        A name with no value is either undeclared, a flag, or a non-required
        argument that was omitted and carries no default — all three are
        caller bugs at the point of reading, so they raise rather than answer
        an empty string."""
        var found = self._values.get(name)
        if found:
            return found.value().copy()
        elif self._flags.get(name):
            raise Error(
                "argparse: '" + name + "' is a flag; read it with flag()"
            )
        else:
            raise Error("argparse: no value for argument '" + name + "'")

    def get_or(self, name: String, var fallback: String) -> String:
        """The value of an option or positional, or `fallback` if it has
        none."""
        var found = self._values.get(name)
        if found:
            return found.value().copy()
        else:
            return fallback^

    def get_optional(self, name: String) -> Optional[String]:
        """The value of an option or positional, or `None` if it has none —
        for a caller that wants to branch rather than supply a fallback."""
        var found = self._values.get(name)
        if found:
            return Optional(found.value().copy())
        else:
            return None

    def get_int(self, name: String) raises -> Int:
        """`get(name)` as an integer, naming the argument on a bad token.

        `atol` raises on its own, but its message names neither the argument
        nor the value — which is the whole difficulty of diagnosing a CLI
        typo."""
        var raw = self.get(name)
        try:
            return atol(raw)
        except:
            raise Error(
                "argparse: '" + name + "' expects an integer, got '" + raw + "'"
            )

    def get_float(self, name: String) raises -> Float64:
        """`get(name)` as a float, naming the argument on a bad token."""
        var raw = self.get(name)
        try:
            return atof(raw)
        except:
            raise Error(
                "argparse: '" + name + "' expects a number, got '" + raw + "'"
            )

    def get_bool(self, name: String) raises -> Bool:
        """`get(name)` as a bool — see `parse_bool` for the accepted
        spellings. For an argument declared with `flag()`, use `flag()`."""
        var raw = self.get(name)
        try:
            return parse_bool(raw)
        except:
            raise Error(
                "argparse: '"
                + name
                + "' expects a bool ('true'/'false'/'1'/'0'), got '"
                + raw
                + "'"
            )

    def flag(self, name: String) raises -> Bool:
        """Whether a declared flag appeared. Raises for a name that was not
        declared as one, so a misspelled flag cannot read as "absent"."""
        var found = self._flags.get(name)
        if found:
            return found.value()
        else:
            raise Error("argparse: '" + name + "' is not a declared flag")

    def supplied(self, name: String) -> Bool:
        """Whether argv mentioned this argument, as opposed to it taking its
        default."""
        var found = self._supplied.get(name)
        if found:
            return found.value()
        else:
            return False

    def trailing(self) -> List[String]:
        """The tokens past the last declared positional — empty unless a
        `trailing()` argument was declared."""
        return self._trailing.copy()


# ---------------------------------------------------------------------------
# ArgumentParser
# ---------------------------------------------------------------------------


def _is_option_token(token: String) -> Bool:
    """Whether `token` is spelled as an option: a leading dash and something
    after it. A lone `-` is a positional (the conventional stdin spelling);
    `--` is the terminator and is checked before this."""
    return token.startswith("-") and token.byte_length() > 1


def _pad_to(var text: String, width: Int) -> String:
    """`text` padded with spaces to at least `width` bytes — the whole of the
    column alignment `help_text` needs, and less than a dependency on a
    formatting facility."""
    for _ in range(width - text.byte_length()):
        text += " "
    return text^


struct ArgumentParser(Copyable, Movable):
    """A command-line grammar: register arguments, then run it over `argv`.

    Registration order is the whole of the parser's state, and it is load
    bearing twice over — positionals are matched in it, and `usage()` /
    `help_text()` render in it. There is no reordering pass and no
    alphabetisation: what a plan author declares first is what a user reads
    first.
    """

    var prog: String
    var description: String
    var _specs: List[ArgSpec]

    def __init__(
        out self,
        var prog: String = String("prog"),
        *,
        var description: String = String(),
    ):
        self.prog = prog^
        self.description = description^
        self._specs = List[ArgSpec]()

    # -- registration -------------------------------------------------------

    def _declare(mut self, var spec: ArgSpec) raises:
        """Append `spec`, rejecting a name or short alias already taken and a
        second trailing argument.

        Every one of those is a declaration bug that would otherwise surface
        as a *parse* bug: a duplicate name makes one of the two declarations
        unreachable, and a second trailing argument can never receive a token
        because the first one consumes everything."""
        for ref existing in self._specs:
            if existing.name == spec.name:
                raise Error("argparse: '" + spec.name + "' is already declared")
            if spec.short.byte_length() > 0 and existing.short == spec.short:
                raise Error(
                    "argparse: short option '-"
                    + spec.short
                    + "' is already taken by '--"
                    + existing.name
                    + "'"
                )
            if (
                spec.kind == ArgSpec.TRAILING
                and existing.kind == ArgSpec.TRAILING
            ):
                raise Error(
                    "argparse: '"
                    + existing.name
                    + "' already collects the trailing arguments"
                )
        self._specs.append(spec^)

    def option(
        mut self,
        var name: String,
        *,
        var short: String = String(),
        var help: String = String(),
        var default: Optional[String] = None,
        var metavar: String = String(),
        required: Bool = True,
    ) raises:
        """Declare `--name VALUE`. Required unless given a `default` or
        `required=False` — see the module docstring for why that is the
        opposite of Python's `argparse`."""
        self._declare(
            ArgSpec(
                name=name^,
                kind=ArgSpec.OPTION,
                short=short^,
                help=help^,
                default=default^,
                metavar=metavar^,
                required=required,
            )
        )

    def flag(
        mut self,
        var name: String,
        *,
        var short: String = String(),
        var help: String = String(),
        short_circuit: Bool = False,
    ) raises:
        """Declare `--name` as a valueless boolean, absent-by-default.

        `short_circuit=True` marks it as meaning "do not run the program"
        — `--describe`, `--version` — which suppresses the required-argument
        check exactly as the built-in `--help` does."""
        self._declare(
            ArgSpec(
                name=name^,
                kind=ArgSpec.FLAG,
                short=short^,
                help=help^,
                short_circuit=short_circuit,
            )
        )

    def positional(
        mut self,
        var name: String,
        *,
        var help: String = String(),
        var default: Optional[String] = None,
        required: Bool = True,
    ) raises:
        """Declare an argument matched by position among the non-option
        tokens, in declaration order."""
        self._declare(
            ArgSpec(
                name=name^,
                kind=ArgSpec.POSITIONAL,
                help=help^,
                default=default^,
                required=required,
            )
        )

    def trailing(
        mut self, var name: String, *, var help: String = String()
    ) raises:
        """Declare a sink for every positional token past the last declared
        positional, read back with `ParsedArgs.trailing()`.

        Without one, a surplus positional is an error. That is the point: the
        sink has to be asked for, so a program that does not want passthrough
        arguments gets told about a typo instead of swallowing it."""
        self._declare(ArgSpec(name=name^, kind=ArgSpec.TRAILING, help=help^))

    # -- lookup -------------------------------------------------------------

    def _index_of(self, name: String) -> Int:
        """The declaration index for `name`, or -1. Linear — see the module
        docstring."""
        for i in range(len(self._specs)):
            if self._specs[i].name == name:
                return i
        return -1

    def _resolve(self, head: String) -> Int:
        """The declaration index an option token names, or -1.

        `--name` matches a long name, `-s` matches a short alias; neither form
        matches a positional, so `--src` is unrecognized when `src` was
        declared with `positional()`."""
        var long = head.startswith("--")
        var bare: String
        if long:
            bare = String(head.removeprefix("--"))
        else:
            bare = String(head.removeprefix("-"))
        for i in range(len(self._specs)):
            ref spec = self._specs[i]
            if spec.is_option_like():
                if long:
                    if spec.name == bare:
                        return i
                elif spec.short.byte_length() > 0 and spec.short == bare:
                    return i
        return -1

    def _nth_positional(self, n: Int) -> Int:
        """The declaration index of the `n`-th positional, or -1 when there
        are fewer than `n + 1` of them."""
        var seen = 0
        for i in range(len(self._specs)):
            if self._specs[i].kind == ArgSpec.POSITIONAL:
                if seen == n:
                    return i
                seen += 1
        return -1

    def _has_trailing(self) -> Bool:
        for ref spec in self._specs:
            if spec.kind == ArgSpec.TRAILING:
                return True
        return False

    # -- parsing ------------------------------------------------------------

    def parse(self, args: List[String]) raises -> ParsedArgs:
        """Run this grammar over `args`, which is `argv` **without** the
        program name.

        Taking a `List[String]` rather than reading `argv` itself is what makes
        every off-by-one in here testable without spawning a process; the
        entry point supplies the real tail. The order is: consume tokens
        left to right, then match the collected positionals, then apply
        defaults, then check for missing required arguments — the last step
        skipped when `--help` or a `short_circuit` flag appeared.
        """
        var values = Dict[String, String]()
        var flags = Dict[String, Bool]()
        var supplied = Dict[String, Bool]()
        var trailing = List[String]()
        var positionals = List[String]()
        var help_requested = False
        var short_circuited = False
        var options_ended = False

        for ref spec in self._specs:
            if spec.kind == ArgSpec.FLAG:
                flags[spec.name.copy()] = False

        var i = 0
        while i < len(args):
            if not options_ended and args[i] == "--":
                options_ended = True
                i += 1
            elif not options_ended and _is_option_token(args[i]):
                var head = args[i].copy()
                var inline = Optional[String](None)
                var eq = head.find("=")
                if eq >= 0:
                    var whole = head.copy()
                    head = String(whole[byte=0:eq])
                    inline = Optional(String(whole[byte = eq + 1 :]))

                var idx = self._resolve(head)
                if idx < 0:
                    if head == "--help" or head == "-h":
                        help_requested = True
                        short_circuited = True
                        i += 1
                    else:
                        raise Error(
                            "argparse: unrecognized option '" + head + "'"
                        )
                else:
                    ref spec = self._specs[idx]
                    if spec.kind == ArgSpec.FLAG:
                        if inline:
                            raise Error(
                                "argparse: '--"
                                + spec.name
                                + "' is a flag and takes no value"
                            )
                        flags[spec.name.copy()] = True
                        supplied[spec.name.copy()] = True
                        if spec.short_circuit:
                            short_circuited = True
                        i += 1
                    else:
                        if inline:
                            values[spec.name.copy()] = inline.value().copy()
                            i += 1
                        elif i + 1 < len(args):
                            values[spec.name.copy()] = args[i + 1].copy()
                            i += 2
                        else:
                            raise Error(
                                "argparse: '--"
                                + spec.name
                                + "' requires a value"
                            )
                        supplied[spec.name.copy()] = True
            else:
                positionals.append(args[i].copy())
                i += 1

        var matched = 0
        var has_trailing = self._has_trailing()
        for k in range(len(positionals)):
            var slot = self._nth_positional(matched)
            if slot >= 0:
                values[self._specs[slot].name.copy()] = positionals[k].copy()
                supplied[self._specs[slot].name.copy()] = True
                matched += 1
            elif has_trailing:
                trailing.append(positionals[k].copy())
            else:
                raise Error(
                    "argparse: unexpected positional argument '"
                    + positionals[k]
                    + "'"
                )

        for ref spec in self._specs:
            var was_supplied = Bool(supplied.get(spec.name))
            if spec.kind == ArgSpec.FLAG or spec.kind == ArgSpec.TRAILING:
                pass
            elif not was_supplied:
                if spec.default:
                    values[spec.name.copy()] = spec.default.value().copy()
                elif spec.is_required() and not short_circuited:
                    raise Error(
                        "argparse: missing required argument '"
                        + spec.spelling()
                        + "'"
                    )

        return ParsedArgs(values^, flags^, supplied^, trailing^, help_requested)

    # -- rendering ----------------------------------------------------------

    def usage(self) -> String:
        """The one-line synopsis, in declaration order. Optional arguments are
        bracketed, required ones are not — the convention every man page
        uses."""
        var out = "usage: " + self.prog
        if self._index_of("help") < 0:
            out += " [-h]"
        for ref spec in self._specs:
            if spec.kind == ArgSpec.FLAG:
                out += " [--" + spec.name + "]"
            elif spec.kind == ArgSpec.OPTION:
                var body = "--" + spec.name + " " + spec.placeholder()
                if spec.is_required():
                    out += " " + body
                else:
                    out += " [" + body + "]"
            elif spec.kind == ArgSpec.POSITIONAL:
                if spec.is_required():
                    out += " <" + spec.name + ">"
                else:
                    out += " [<" + spec.name + ">]"
            else:
                out += " [" + spec.name + " ...]"
        return out^

    def _invocation(self, spec: ArgSpec) -> String:
        """The left column of a `help_text` row.

        Options and flags are indented past a four-column `-s, ` slot whether
        or not they have a short alias, so the `--name`s line up whichever
        subset declares one."""
        if spec.is_option_like():
            var head: String
            if spec.short.byte_length() > 0:
                head = "-" + spec.short + ", "
            else:
                head = String("    ")
            head += "--" + spec.name
            if spec.kind == ArgSpec.OPTION:
                head += " " + spec.placeholder()
            return head^
        elif spec.kind == ArgSpec.TRAILING:
            return spec.name + " ..."
        else:
            return spec.name.copy()

    def help_text(self) -> String:
        """The full `--help` page: synopsis, description, then one row per
        argument under `options:` and `arguments:`.

        Every row states its status — `(required)` or `(default: X)` — because
        the whole question a reader brings to `--help` is what they have to
        supply. The two sections are omitted when empty, and the built-in
        `-h, --help` row is omitted when the caller declared its own `help`.
        """
        var out = self.usage() + "\n"
        if self.description.byte_length() > 0:
            out += "\n" + self.description + "\n"

        var width = 0
        var builtin_help = self._index_of("help") < 0
        if builtin_help:
            width = 15  # len("-h, --help") plus the two-space gutter
        for ref spec in self._specs:
            var w = self._invocation(spec).byte_length() + 2
            if w > width:
                width = w

        var options = String()
        var arguments = String()
        if builtin_help:
            options += "  " + _pad_to(String("-h, --help"), width)
            options += "show this help message\n"
        for ref spec in self._specs:
            var row = "  " + _pad_to(self._invocation(spec), width)
            row += spec.help
            if spec.is_required():
                if spec.help.byte_length() > 0:
                    row += " "
                row += "(required)"
            elif spec.default:
                if spec.help.byte_length() > 0:
                    row += " "
                row += "(default: " + spec.default.value() + ")"
            row += "\n"
            if spec.is_option_like():
                options += row
            else:
                arguments += row

        if options.byte_length() > 0:
            out += "\noptions:\n" + options
        if arguments.byte_length() > 0:
            out += "\narguments:\n" + arguments
        return out^
