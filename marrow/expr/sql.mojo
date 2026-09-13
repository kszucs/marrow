"""A SQL front end: a query string in, a `DynRelation` out.

One file, because the three stages are one pipeline and each is only useful to
the next:

    Lexer.scan -> Parser.parse -> Sql.plan

`Sql.plan("SELECT ...", catalog)` is the whole public surface. Everything else
is a type it uses.

**Tags are strings, and an operator\'s tag *is* the operator.** A parsed `a +
b` is `Node("+", a, b)` and `x NOT IN (...)` is `Node("not_in", ...)`, which
is the same shape `RuntimeValue` uses (`marrow/expr/runtime/values.mojo`
switches on a `_tag` for exactly this reason). It costs a string compare where
an enum would compare an integer, and it buys a parse tree that prints as
itself and a planner whose dispatch reads like the SQL it came from — no
`E_BINARY` plus a separate `text` holding `"+"`, and no integer flag field
packing "negated" and "case-insensitive" into one `ival`.

**The parse tree is a value.** `Ast` is parallel lists addressed by index, not
a graph of pointers: a `List` whose element is a `Variant` drops elements when
it grows (CLAUDE.md), and a flat value can be a **comptime parameter**, which
is what a future AOT lane would bind against.

**Everything up to `Parser.parse` is non-raising**, deliberately. The compiler
will not run a raising function at compile time — the metaprogramming manual
lists it beside file I/O and FFI — so `comptime AST = Parser.parse(sql)`
compiles only as long as the lexer and parser stay that way. Errors are
therefore *recorded*: `Ast.error` is the only failure signal before `Sql.plan`,
which is allowed to raise and does.

Clause order is SQL\'s evaluation order, not the written one:

    FROM -> JOIN -> WHERE -> GROUP BY/aggregates -> HAVING -> SELECT
      -> DISTINCT -> ORDER BY -> LIMIT

Three constructs are desugared rather than given nodes of their own, because
the sugar has exactly SQL\'s semantics and the node would not:

- `x IN (a, b)` becomes `x = a OR x = b`. That is not a shortcut around
  `isin`, which decides membership on a 64-bit hash: the chain reproduces
  SQL\'s three-valued rule, under which `x IN (1, NULL)` is NULL rather than
  false. A NULL in the list is lifted out of the chain, since `x = NULL` is
  NULL for every row.
- `GREATEST`/`LEAST` become `coalesce(extremum, a, b)`, because SQL\'s extrema
  **skip** nulls while `maximum`/`minimum` propagate them.
- `SELECT DISTINCT` becomes an aggregate keyed by every output column, marrow
  having no `Distinct` node (`backlog.md` item 7).
"""

from ..dtypes import (
    DynType,
    bool_,
    date32,
    float32,
    float64,
    int16,
    int32,
    int64,
    int8,
    microsecond,
    string,
    timestamp,
    uint16,
    uint32,
    uint64,
    uint8,
)
from ..kernels.join import (
    JOIN_FULL,
    JOIN_INNER,
    JOIN_LEFT,
    JOIN_RIGHT,
    JoinKind,
)
from ..scalars import (
    BoolScalar,
    DynScalar,
    Float64Scalar,
    Int32Scalar,
    Int64Scalar,
    NullScalar,
    StringScalar,
)
from ..schema import Schema
from ..tabular import RecordBatch
from .builders import table
from .logical import DynRelation, DynValue
from .runtime.aggregates import RuntimeAggregate
from .runtime.values import (
    RuntimeValue,
    abs,
    add,
    and_,
    case_when,
    cast,
    ceil,
    char_length,
    coalesce,
    column,
    contains,
    cos,
    array_length,
    ascii,
    date_trunc,
    day_name,
    day_of_week,
    day_of_year,
    epoch,
    exp2,
    is_inf,
    is_nan,
    iso_year,
    last_day,
    log1p,
    month_name,
    split_part,
    trim_chars,
    week,
    day,
    endswith,
    eq,
    exp,
    fill_null,
    floor,
    ge,
    gt,
    hour,
    ilike,
    is_null,
    is_valid,
    le,
    left,
    length,
    like,
    literal,
    ln,
    log10,
    log2,
    lower,
    lpad,
    lstrip,
    lt,
    maximum,
    minimum,
    minute,
    mod,
    month,
    mul,
    ne,
    neg,
    not_,
    nullif,
    or_,
    position,
    pow,
    quarter,
    repeat,
    replace,
    reverse,
    right,
    round,
    rpad,
    rstrip,
    second,
    sign,
    sin,
    sqrt,
    startswith,
    strip,
    sub,
    substr,
    trunc,
    truediv,
    upper,
    xor,
    year,
)


struct Ascii:
    """Byte classification and case folding, all of it non-raising.

    SQL is an ASCII language even when its data is not, so every question the
    lexer asks about a byte is answerable here — and none of the obvious
    stdlib spellings can be used: `chr`, `StringSlice(from_utf8=...)` and
    `String.upper()` all raise, and a raising function cannot run at compile
    time.
    """

    comptime UPPERCASE = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    """Folded through by index, since every other spelling raises."""

    @staticmethod
    def is_digit(b: Int) -> Bool:
        return b >= 48 and b <= 57

    @staticmethod
    def is_alpha(b: Int) -> Bool:
        return (b >= 97 and b <= 122) or (b >= 65 and b <= 90) or b == 95

    @staticmethod
    def is_word_byte(b: Int) -> Bool:
        return Ascii.is_alpha(b) or Ascii.is_digit(b)

    @staticmethod
    def is_space(b: Int) -> Bool:
        return b == 32 or b == 9 or b == 10 or b == 13

    @staticmethod
    def to_int(text: String) -> Int:
        """`text` as an `Int`, hand-rolled because `Int(String)` raises.

        Only ever fed a digit run the lexer already validated, so there is no
        malformed case to report; overflow wraps, which is what `LIMIT
        99999999999999999999` deserves in a prototype.
        """
        var value = 0
        for i in range(text.byte_length()):
            var b = Int(text.as_bytes()[i])
            if Ascii.is_digit(b):
                value = value * 10 + (b - 48)
        return value

    @staticmethod
    def lookup(table: StringSlice, key: String) -> String:
        """The value for `key` in a space-separated `key:value` table, or "".

        The table is a string for the same reason `contains_word` takes one:
        Mojo has no module-level `List`, and a `comptime` one must be
        `materialize`d back at every use.
        """
        var padded = String(" ") + table + " "
        var probe = String(" ") + key + ":"
        var at = padded.find(probe)
        if at < 0:
            return String("")
        var start = at + probe.byte_length()
        var end = start
        while end < padded.byte_length() and padded.as_bytes()[end] != 32:
            end += 1
        return String(padded[byte=start:end])

    @staticmethod
    def contains_word(words: StringSlice, word: String) -> Bool:
        """Whether `word` is one of the space-separated `words`.

        A string rather than a `List` of them: a module-level `List` cannot
        exist in Mojo, and a `comptime` one has to be `materialize`d back at
        every use. One padded `find` says the same thing without either.
        """
        return (String(" ") + words + " ").find(" " + word + " ") >= 0

    @staticmethod
    def upper(s: String) -> String:
        """`s` with a-z folded to A-Z, every other byte left alone.

        Hand-rolled rather than `String.upper()` because this runs at comptime and
        must not raise; ASCII is also the whole of SQL's case-insensitivity rule.

        Copied in **runs** rather than a byte at a time. `s[byte=i]` aborts unless
        `i` is a codepoint boundary, and the bytes of a multi-byte character are
        not — but a lowercase ASCII letter always is, so cutting only at the
        letters being folded keeps every slice legal whatever else `s` holds.
        """
        var out = String("")
        var run = 0
        for i in range(s.byte_length()):
            var b = Int(s.as_bytes()[i])
            if b >= 97 and b <= 122:
                out += String(s[byte=run:i])
                out += String(Ascii.UPPERCASE[byte=b - 97])
                run = i + 1
        out += String(s[byte = run : s.byte_length()])
        return out^


struct Token(Copyable, Movable, Writable):
    """One lexeme, plus where it started so an error can point at it."""

    var kind: String

    var text: String
    """As written: the identifier, the number, the *decoded* string body, the
    punctuation, or an error message."""

    var upper: String
    """`Ascii.upper(text)` for a `WORD`, empty otherwise. Precomputed because
    every keyword test in the parser would otherwise fold the same bytes."""

    var pos: Int

    var is_float: Bool
    """Set on a `NUMBER` carrying a `.` or an exponent, so the planner picks
    `float64` over `int64` without rescanning the text."""

    var quoted: Bool
    """Set on a `WORD` that arrived double-quoted, which makes it an identifier
    even when it spells a keyword."""

    def __init__(
        out self,
        var kind: String,
        var text: String,
        pos: Int,
        is_float: Bool = False,
        quoted: Bool = False,
    ):
        self.upper = Ascii.upper(text) if kind == "word" else String("")
        self.kind = kind^
        self.text = text^
        self.pos = pos
        self.is_float = is_float
        self.quoted = quoted

    def is_word(self, keyword: StringSlice) -> Bool:
        """True for an unquoted word spelling `keyword` in any case.

        **`keyword` must be written upper-case.** The comparison is against
        the pre-folded `upper`, so folding the argument too would repeat that
        work on every keyword test in the parser — and every call site is a
        literal, where writing `"SELECT"` costs nothing.
        """
        return self.kind == "word" and not self.quoted and self.upper == keyword

    def is_any_of(self, keywords: StringSlice) -> Bool:
        """True for an unquoted word appearing in the space-separated
        `keywords`, all of which must be upper-case."""
        if self.kind != "word" or self.quoted:
            return False
        return Ascii.contains_word(keywords, self.upper)

    def is_punct(self, symbol: StringSlice) -> Bool:
        return self.kind == "punct" and self.text == symbol

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self.text)


comptime NONE = -1
"""What an absent edge holds. Never a valid node index."""


struct Node(Copyable, Movable):
    """One expression node. A flat representation pays for its uniformity with
    unused slots — an `"int"` carries `value` and nothing else."""

    var kind: String
    """The tag, and for an operator it *is* the operator: `"+"`, `"<>"`,
    `"and"`, `"not_in"`, `"not_ilike"`. Nothing here carries a separate flag
    saying whether a predicate was negated."""

    var a: Int
    var b: Int
    var c: Int

    var text: String
    """A column or function name, a literal as written, a cast target."""

    var qualifier: String
    """The table qualifier of a column or of `t.*`, or empty."""

    var value: Int
    """An integer literal's value, and a boolean's 0 or 1. Kept beside `text`
    because `LIMIT` and ordinals need it during the parse, where `Int(String)`
    cannot be called."""

    var distinct: Bool
    """`COUNT(DISTINCT x)`."""

    var kids_start: Int
    var kids_len: Int

    def qualified_name(self) -> String:
        """`"e.did"` for a qualified column, `"did"` for a bare one."""
        return (
            self.text.copy() if self.qualifier
            == "" else self.qualifier + "." + self.text
        )

    def __init__(
        out self,
        var kind: String,
        a: Int = NONE,
        b: Int = NONE,
        c: Int = NONE,
        var text: String = String(""),
        var qualifier: String = String(""),
        value: Int = 0,
        distinct: Bool = False,
        kids_start: Int = 0,
        kids_len: Int = 0,
    ):
        self.kind = kind^
        self.a = a
        self.b = b
        self.c = c
        self.text = text^
        self.qualifier = qualifier^
        self.value = value
        self.distinct = distinct
        self.kids_start = kids_start
        self.kids_len = kids_len


@fieldwise_init
struct Join(Copyable, Movable):
    """One joined table and its equality keys.

    Only equijoins: `marrow.expr`'s `Join` takes two lists of *positional* key
    indices, so `ON a.x = b.y AND a.p = b.q` is the whole grammar it can
    express. A non-equi predicate parses and then fails in the planner, which
    is the honest place for it to fail.
    """

    var kind: String
    var table: String
    var alias_name: String
    var left_keys: List[String]
    var right_keys: List[String]

    def to_kind(self) raises -> JoinKind:
        if self.kind == "inner":
            return JOIN_INNER
        if self.kind == "left":
            return JOIN_LEFT
        if self.kind == "right":
            return JOIN_RIGHT
        if self.kind == "full":
            return JOIN_FULL
        raise Error("sql: unsupported join kind")


@fieldwise_init
struct Item(Copyable, Movable):
    """One entry of the `SELECT` list."""

    var node: Int
    var alias_name: String
    """Empty means "derive the name from the expression"."""


@fieldwise_init
struct OrderKey(Copyable, Movable):
    """One entry of `ORDER BY`."""

    var node: Int
    var ascending: Bool
    var nulls_first: Bool
    var nulls_explicit: Bool
    """Whether the query *wrote* `NULLS FIRST`/`LAST`. `Sort` carries one flag
    for every key, so a conflict only matters when it was asked for; an
    unstated default must never manufacture one."""


struct Select(Copyable, Movable):
    """One `SELECT` statement, with every clause resolved to node indices.

    `Item` and `OrderKey` exist so that a clause is **one** list. The four
    parallel lists `ORDER BY` used to need could disagree in length, and only
    the code appending to them knew they were meant to line up.
    """

    var distinct: Bool
    var items: List[Item]
    var table: String
    var table_alias: String
    var joins: List[Join]
    var predicate: Int
    """The `WHERE` clause, or `NONE`. Not named `where`: that is a reserved
    word, and while the compiler accepts it as a field the formatter cannot
    parse it."""
    var group_by: List[Int]
    var having: Int
    var order: List[OrderKey]
    var limit: Int
    """-1 for no `LIMIT`."""
    var offset: Int

    def __init__(out self):
        """Every clause absent. Not `@fieldwise_init`: Mojo has no field
        defaults, so that would generate an eleven-argument constructor and
        leave this one to be written anyway."""
        self.distinct = False
        self.items = List[Item]()
        self.table = String("")
        self.table_alias = String("")
        self.joins = List[Join]()
        self.predicate = NONE
        self.group_by = List[Int]()
        self.having = NONE
        self.order = List[OrderKey]()
        self.limit = -1
        self.offset = 0


struct Ast(Copyable, Movable):
    """Every node of one parsed statement, plus the first error if there was
    one.

    `error` being non-empty is the only failure signal — the parser cannot
    raise, so a caller must look. `planner.plan` looks first.
    """

    var nodes: List[Node]
    var kids: List[Int]
    var select: Select
    var error: String
    var error_pos: Int

    def __init__(out self):
        self.nodes = List[Node]()
        self.kids = List[Int]()
        self.select = Select()
        self.error = String("")
        self.error_pos = 0

    def add(mut self, var node: Node) -> Int:
        """Append `node` and answer its index."""
        self.nodes.append(node^)
        return len(self.nodes) - 1

    def add_kids(mut self, kids: List[Int]) -> Int:
        """Append a variadic edge list and answer where it starts."""
        var start = len(self.kids)
        for k in kids:
            self.kids.append(k)
        return start

    def ok(self) -> Bool:
        return self.error == ""

    def kid(self, node: Node, i: Int) -> Int:
        """The `i`th variadic child of `node`."""
        return self.kids[node.kids_start + i]


comptime BP_OR = 1
comptime BP_AND = 2
comptime BP_NOT = 3
comptime BP_COMPARE = 4
comptime BP_CONCAT = 5
comptime BP_ADD = 6
comptime BP_MUL = 7
comptime BP_NEG = 8


struct Parser(Copyable, Movable):
    var toks: List[Token]
    var pos: Int
    var ast: Ast

    def __init__(out self, var sql: String):
        self.toks = Parser.scan(sql)
        self.pos = 0
        self.ast = Ast()

    @staticmethod
    def scan(src: String) -> List[Token]:
        """The source as tokens, always terminated by one `"eof"`.

        A lexing failure appends one `"error"` token and stops: a prototype
        that guesses at a malformed query is worse than one that names the
        byte it choked on.
        """
        var out = List[Token]()
        var n = src.byte_length()
        var i = 0
        while i < n:
            var b = Int(src.as_bytes()[i])
            if Ascii.is_space(b):
                i += 1
            elif b == 45 and i + 1 < n and Int(src.as_bytes()[i + 1]) == 45:
                # `-- line comment`
                while i < n and Int(src.as_bytes()[i]) != 10:
                    i += 1
            elif b == 47 and i + 1 < n and Int(src.as_bytes()[i + 1]) == 42:
                # `/* block comment */`
                var closed = False
                i += 2
                while i + 1 < n:
                    if (
                        Int(src.as_bytes()[i]) == 42
                        and Int(src.as_bytes()[i + 1]) == 47
                    ):
                        i += 2
                        closed = True
                        break
                    i += 1
                if not closed:
                    out.append(Token("error", "unterminated block comment", i))
                    break
            elif Ascii.is_alpha(b):
                var start = i
                while i < n and Ascii.is_word_byte(Int(src.as_bytes()[i])):
                    i += 1
                out.append(Token("word", String(src[byte=start:i]), start))
            elif b == 34:
                # `"quoted identifier"`, with `""` as the escape
                var start = i
                i += 1
                var run = i
                var word = String("")
                var closed = False
                while i < n:
                    if Int(src.as_bytes()[i]) == 34:
                        word += String(src[byte=run:i])
                        i += 1
                        if i < n and Int(src.as_bytes()[i]) == 34:
                            word += '"'
                            i += 1
                            run = i
                        else:
                            closed = True
                            break
                    else:
                        i += 1
                if closed:
                    out.append(Token("word", word^, start, quoted=True))
                else:
                    out.append(
                        Token("error", "unterminated quoted identifier", start)
                    )
                    break
            elif b == 39:
                # `'string literal'`, with `''` as the escape
                var start = i
                i += 1
                var run = i
                var body = String("")
                var closed = False
                while i < n:
                    if Int(src.as_bytes()[i]) == 39:
                        body += String(src[byte=run:i])
                        i += 1
                        if i < n and Int(src.as_bytes()[i]) == 39:
                            body += "'"
                            i += 1
                            run = i
                        else:
                            closed = True
                            break
                    else:
                        i += 1
                if closed:
                    out.append(Token("string", body^, start))
                else:
                    out.append(
                        Token("error", "unterminated string literal", start)
                    )
                    break
            elif Ascii.is_digit(b) or (
                b == 46
                and i + 1 < n
                and Ascii.is_digit(Int(src.as_bytes()[i + 1]))
            ):
                var start = i
                var is_float = False
                while i < n:
                    var c = Int(src.as_bytes()[i])
                    if Ascii.is_digit(c):
                        i += 1
                    elif c == 46 and not is_float:
                        is_float = True
                        i += 1
                    elif (c == 101 or c == 69) and i + 1 < n:
                        # An exponent, but only when digits (or a signed pair of
                        # them) follow — otherwise `1e` would eat the `e` of a
                        # following word.
                        var nxt = Int(src.as_bytes()[i + 1])
                        var signed = (nxt == 43 or nxt == 45) and i + 2 < n
                        var exponent = Ascii.is_digit(nxt) or (
                            signed
                            and Ascii.is_digit(Int(src.as_bytes()[i + 2]))
                        )
                        if exponent:
                            is_float = True
                            i += 1
                            if signed:
                                i += 1
                        else:
                            break
                    else:
                        break
                out.append(
                    Token(
                        "number",
                        String(src[byte=start:i]),
                        start,
                        is_float=is_float,
                    )
                )
            else:
                var start = i
                var two = String("")
                if i + 1 < n:
                    two = String(src[byte=i]) + String(src[byte=i + 1])
                var is_two = (
                    two == "<="
                    or two == ">="
                    or two == "<>"
                    or two == "!="
                    or two == "||"
                )
                var is_one = (
                    b == 40  # (
                    or b == 41  # )
                    or b == 44  # ,
                    or b == 42  # *
                    or b == 43  # +
                    or b == 45  # -
                    or b == 47  # /
                    or b == 37  # %
                    or b == 60  # <
                    or b == 62  # >
                    or b == 61  # =
                    or b == 46  # .
                )
                if is_two:
                    out.append(Token("punct", two^, start))
                    i += 2
                elif is_one:
                    out.append(Token("punct", String(src[byte=i]), start))
                    i += 1
                else:
                    out.append(
                        Token(
                            "error",
                            "unexpected character '"
                            + String(src[byte=i])
                            + "'",
                            start,
                        )
                    )
                    break
        out.append(Token("eof", "", n))
        return out^

    # -- cursor -------------------------------------------------------------

    def peek(ref self, ahead: Int = 0) -> ref[self.toks[self.pos]] Token:
        """The token `ahead` of the cursor, clamped to the `"eof"` that always
        ends the list — so a `NOT` at the very end has something to look at.

        One accessor with an offset, rather than `at`/`peek`/`lookahead`: the
        only reason there were three is that the origin has to be spelled, and
        it is the same origin for every element.
        """
        var last = len(self.toks) - 1
        var index = self.pos + ahead
        return self.toks[index if index < last else last]

    def advance(mut self):
        if self.pos < len(self.toks) - 1:
            self.pos += 1

    def at_word(self, keyword: StringSlice) -> Bool:
        return self.peek().is_word(keyword)

    def at_punct(self, symbol: StringSlice) -> Bool:
        return self.peek().is_punct(symbol)

    def at_clause_word(self) -> Bool:
        """Whether the cursor is on a word that continues the statement rather
        than aliasing the item before it — the reason `SELECT a FROM t` does
        not read `FROM` as an alias for `a`."""
        return self.peek().is_any_of(
            "FROM WHERE GROUP HAVING ORDER LIMIT OFFSET JOIN INNER LEFT RIGHT"
            " FULL CROSS ON AND OR AS ASC DESC NULLS UNION EXCEPT INTERSECT"
            " WHEN THEN ELSE END"
        )

    def take_word(mut self, keyword: StringSlice) -> Bool:
        """Consume `keyword` if it is next, and say whether it was."""
        if self.at_word(keyword):
            self.advance()
            return True
        return False

    def take_punct(mut self, symbol: StringSlice) -> Bool:
        if self.at_punct(symbol):
            self.advance()
            return True
        return False

    def fail(mut self, var message: String):
        """Record the first error; later ones are noise from unwinding."""
        if self.ast.ok():
            self.ast.error = message^
            self.ast.error_pos = self.peek().pos

    @staticmethod
    def is_comparison(symbol: String) -> Bool:
        return (
            symbol == "="
            or symbol == "<>"
            or symbol == "!="
            or symbol == "<"
            or symbol == "<="
            or symbol == ">"
            or symbol == ">="
        )

    def emit(mut self, var node: Node) -> Int:
        """Add `node`, or `NONE` if the parse has already failed.

        Every production that builds a node has just called something that
        might have failed, so without this each one ends in the same
        four-line ternary. One name for the check keeps the grammar readable.
        """
        return NONE if not self.ast.ok() else self.ast.add(node^)

    def expect_punct(mut self, symbol: StringSlice) -> Bool:
        if self.take_punct(symbol):
            return True
        self.fail(
            "expected '"
            + String(symbol)
            + "', found '"
            + self.peek().text
            + "'"
        )
        return False

    def expect_word(mut self, keyword: StringSlice) -> Bool:
        if self.take_word(keyword):
            return True
        self.fail(
            "expected " + String(keyword) + ", found '" + self.peek().text + "'"
        )
        return False

    def identifier(mut self) -> String:
        """The next token as a name, or "" with an error recorded."""
        if self.peek().kind == "word":
            var name = self.peek().text.copy()
            self.advance()
            return name^
        self.fail("expected a name, found '" + self.peek().text + "'")
        return String("")

    # -- expressions --------------------------------------------------------

    def binding_power(self) -> Int:
        """Left binding power of the token at the cursor; 0 if it does not
        continue an expression.

        `NOT` answers `BP_COMPARE` only in front of the predicate it negates,
        so `a NOT IN (1)` binds here while `NOT a` stays a prefix.
        """
        ref tok = self.peek()
        if tok.kind == "punct":
            if tok.text == "||":
                return BP_CONCAT
            if tok.text == "+" or tok.text == "-":
                return BP_ADD
            if tok.text == "*" or tok.text == "/" or tok.text == "%":
                return BP_MUL
            if Parser.is_comparison(tok.text):
                return BP_COMPARE
            return 0
        if tok.kind != "word" or tok.quoted:
            return 0
        if tok.upper == "OR":
            return BP_OR
        if tok.upper == "AND":
            return BP_AND
        if (
            tok.upper == "IS"
            or tok.upper == "IN"
            or tok.upper == "LIKE"
            or tok.upper == "ILIKE"
            or tok.upper == "BETWEEN"
        ):
            return BP_COMPARE
        if tok.upper == "NOT":
            ref after = self.peek(1)
            var negatable = (
                after.is_word("IN")
                or after.is_word("LIKE")
                or after.is_word("ILIKE")
                or after.is_word("BETWEEN")
            )
            return BP_COMPARE if negatable else 0
        return 0

    def expression(mut self, min_bp: Int = 0) -> Int:
        """One expression, binding no looser than `min_bp`."""
        var lhs = self.prefix()
        while self.ast.ok():
            var bp = self.binding_power()
            if bp == 0 or bp < min_bp:
                break
            lhs = self.infix(lhs, bp)
        return lhs

    def prefix(mut self) -> Int:
        if self.take_punct("-"):
            var operand = self.expression(BP_NEG)
            return self.emit(Node("neg", a=operand))
        if self.take_punct("+"):
            return self.expression(BP_NEG)
        if self.take_word("NOT"):
            var operand = self.expression(BP_NOT)
            return self.emit(Node("not", a=operand))
        return self.primary()

    def infix(mut self, lhs: Int, bp: Int) -> Int:
        """Consume the operator at the cursor and everything it takes."""
        var negated = 1 if self.take_word("NOT") else 0
        # A copy, not a borrow: every arm below advances the cursor, which
        # invalidates an interior reference into `self.toks`.
        var tok = self.peek().copy()
        if tok.kind == "punct":
            # `!=` is spelled away here so the tree carries one inequality and
            # everything downstream matches one tag.
            var op = String("<>") if tok.text == "!=" else tok.text.copy()
            self.advance()
            var rhs = self.expression(bp + 1)
            return self.emit(Node(op^, a=lhs, b=rhs))
        if tok.is_word("IS"):
            self.advance()
            var is_not = 1 if self.take_word("NOT") else 0
            if not self.expect_word("NULL"):
                return NONE
            return self.ast.add(
                Node("is_not_null" if is_not else "is_null", a=lhs)
            )
        if tok.is_word("IN"):
            self.advance()
            return self.in_list(lhs, negated)
        if tok.is_word("LIKE") or tok.is_word("ILIKE"):
            var verb = String("ilike") if tok.is_word("ILIKE") else String(
                "like"
            )
            self.advance()
            var pattern = self.expression(bp + 1)
            return self.emit(
                Node(("not_" + verb) if negated else verb^, a=lhs, b=pattern)
            )
        if tok.is_word("BETWEEN"):
            self.advance()
            # Above `AND`, so the bound stops at the `AND` that closes the
            # construct instead of consuming it as a conjunction.
            var lo = self.expression(BP_CONCAT)
            if not self.expect_word("AND"):
                return NONE
            var hi = self.expression(BP_CONCAT)
            return self.emit(
                Node("not_between" if negated else "between", a=lhs, b=lo, c=hi)
            )
        # `AND` / `OR`, the only remaining word operators. Lower-cased so
        # every tag in the tree is written the way it is matched.
        var op = String("and") if tok.is_word("AND") else String("or")
        self.advance()
        var rhs = self.expression(bp + 1)
        return self.emit(Node(op^, a=lhs, b=rhs))

    def in_list(mut self, lhs: Int, negated: Int) -> Int:
        if not self.expect_punct("("):
            return NONE
        var items = self.comma_list(")")
        if not self.expect_punct(")"):
            return NONE
        var start = self.ast.add_kids(items)
        return self.ast.add(
            Node(
                "not_in" if negated else "in",
                a=lhs,
                kids_start=start,
                kids_len=len(items),
            )
        )

    def comma_list(mut self, terminator: StringSlice) -> List[Int]:
        """Zero or more comma-separated expressions, stopping *before*
        `terminator`."""
        var items = List[Int]()
        if self.at_punct(terminator):
            return items^
        while True:
            var item = self.expression()
            if not self.ast.ok():
                return items^
            items.append(item)
            if not self.take_punct(","):
                break
        return items^

    def primary(mut self) -> Int:
        var tok = self.peek().copy()
        if tok.kind == "error":
            self.fail(tok.text.copy())
            return NONE
        if tok.kind == "number":
            self.advance()
            if tok.is_float:
                # The text, not a value: `Float64(String)` raises, and the
                # planner is the layer allowed to.
                return self.ast.add(Node("float", text=tok.text.copy()))
            return self.ast.add(
                Node(
                    "int",
                    text=tok.text.copy(),
                    value=Ascii.to_int(tok.text),
                )
            )
        if tok.kind == "string":
            self.advance()
            return self.ast.add(Node("string", text=tok.text.copy()))
        if tok.is_punct("("):
            self.advance()
            var inner = self.expression()
            return NONE if not self.expect_punct(")") else inner
        if tok.is_punct("*"):
            self.advance()
            return self.ast.add(Node("star"))
        if tok.kind == "word":
            if tok.is_word("NULL"):
                self.advance()
                return self.ast.add(Node("null"))
            if tok.is_word("TRUE") or tok.is_word("FALSE"):
                var truth = 1 if tok.is_word("TRUE") else 0
                self.advance()
                return self.ast.add(Node("bool", value=truth))
            if tok.is_word("CASE"):
                return self.case_expr()
            if tok.is_word("CAST"):
                return self.cast_expr()
            if self.peek(1).is_punct("("):
                return self.call_expr()
            return self.column_expr()
        self.fail("unexpected '" + tok.text + "'")
        return NONE

    def column_expr(mut self) -> Int:
        var first = self.identifier()
        if not self.ast.ok():
            return NONE
        if not self.take_punct("."):
            return self.ast.add(Node("column", text=first^))
        if self.take_punct("*"):
            return self.ast.add(Node("star", qualifier=first^))
        var name = self.identifier()
        return self.emit(Node("column", text=name^, qualifier=first^))

    def call_expr(mut self) -> Int:
        var name = self.peek().upper.copy()
        self.advance()
        if not self.expect_punct("("):
            return NONE
        var distinct = self.take_word("DISTINCT")
        var args = self.comma_list(")")
        if not self.expect_punct(")"):
            return NONE
        var start = self.ast.add_kids(args)
        return self.ast.add(
            Node(
                "func",
                text=name^,
                distinct=distinct,
                kids_start=start,
                kids_len=len(args),
            )
        )

    def cast_expr(mut self) -> Int:
        self.advance()  # CAST
        if not self.expect_punct("("):
            return NONE
        var operand = self.expression()
        if not self.expect_word("AS"):
            return NONE
        var type_name = self.type_name()
        if not self.ast.ok() or not self.expect_punct(")"):
            return NONE
        return self.ast.add(Node("cast", a=operand, text=type_name^))

    def type_name(mut self) -> String:
        """A SQL type as written, upper-cased, with any `(p, s)` kept.

        Text rather than a resolved `DataType`: that is marrow's vocabulary,
        not SQL's, and this module does not import it.
        """
        if self.peek().kind != "word":
            self.fail("expected a type name, found '" + self.peek().text + "'")
            return String("")
        var name = self.peek().upper.copy()
        self.advance()
        # A second word that belongs to the type — `DOUBLE PRECISION`,
        # `TIMESTAMP WITH TIME ZONE` — rather than starting the next clause.
        while self.peek().is_any_of("PRECISION VARYING WITHOUT WITH TIME ZONE"):
            name += " " + self.peek().upper
            self.advance()
        if self.take_punct("("):
            name += "("
            while not self.at_punct(")") and self.peek().kind != "eof":
                name += self.peek().text
                self.advance()
            if not self.expect_punct(")"):
                return String("")
            name += ")"
        return name^

    def case_expr(mut self) -> Int:
        self.advance()  # CASE
        var operand = NONE
        if not self.at_word("WHEN"):
            # `CASE x WHEN 1 THEN ...` — the simple form. Each arm becomes
            # `x = <arm>`, so the planner only ever sees the searched shape.
            operand = self.expression()
            if not self.ast.ok():
                return NONE
        var arms = List[Int]()
        while self.take_word("WHEN"):
            var condition = self.expression()
            if not self.ast.ok():
                return NONE
            if operand != NONE:
                condition = self.ast.add(Node("=", a=operand, b=condition))
            if not self.expect_word("THEN"):
                return NONE
            var result = self.expression()
            if not self.ast.ok():
                return NONE
            arms.append(condition)
            arms.append(result)
        if len(arms) == 0:
            self.fail("CASE needs at least one WHEN")
            return NONE
        var otherwise = NONE
        if self.take_word("ELSE"):
            otherwise = self.expression()
            if not self.ast.ok():
                return NONE
        if not self.expect_word("END"):
            return NONE
        var start = self.ast.add_kids(arms)
        return self.ast.add(
            Node("case", a=otherwise, kids_start=start, kids_len=len(arms))
        )

    # -- clauses ------------------------------------------------------------

    @staticmethod
    def parse(var sql: String) -> Ast:
        """`sql` as a `Ast`. Check `.ok()` — this cannot raise."""
        var parser = Parser(sql^)
        parser.statement()
        return parser.ast.copy()

    def statement(mut self):
        if not self.expect_word("SELECT"):
            return
        self.ast.select.distinct = self.take_word("DISTINCT")
        self.select_list()
        if self.ast.ok() and self.take_word("FROM"):
            self.from_clause()
        if self.ast.ok() and self.take_word("WHERE"):
            self.ast.select.predicate = self.expression()
        if self.ast.ok() and self.take_word("GROUP"):
            if self.expect_word("BY"):
                self.ast.select.group_by = self.comma_list("")
        if self.ast.ok() and self.take_word("HAVING"):
            self.ast.select.having = self.expression()
        if self.ast.ok() and self.take_word("ORDER"):
            if self.expect_word("BY"):
                self.order_by()
        if self.ast.ok() and self.take_word("LIMIT"):
            self.ast.select.limit = self.whole_number("LIMIT")
        if self.ast.ok() and self.take_word("OFFSET"):
            self.ast.select.offset = self.whole_number("OFFSET")
        if not self.ast.ok():
            return
        if self.peek().kind == "error":
            self.fail(self.peek().text.copy())
        elif self.peek().kind != "eof":
            self.fail("unexpected trailing '" + self.peek().text + "'")

    def whole_number(mut self, clause: StringSlice) -> Int:
        if self.peek().kind != "number" or self.peek().is_float:
            # `_parse_digits` drops every non-digit byte, so without the
            # `is_float` guard `LIMIT 1.5` would silently become `LIMIT 15`.
            self.fail(String(clause) + " needs a whole number")
            return 0
        var value = Ascii.to_int(self.peek().text)
        self.advance()
        return value

    def select_list(mut self):
        while True:
            var item = self.expression()
            if not self.ast.ok():
                return
            self.ast.select.items.append(Item(item, self.item_alias()))
            if not self.ast.ok() or not self.take_punct(","):
                return

    def item_alias(mut self) -> String:
        """`AS name`, a bare trailing name, or "" to derive one."""
        if self.take_word("AS"):
            return self.identifier()
        if self.peek().kind == "word" and not self.at_clause_word():
            var name = self.peek().text.copy()
            self.advance()
            return name^
        return String("")

    def from_clause(mut self):
        self.ast.select.table = self.identifier()
        if not self.ast.ok():
            return
        self.ast.select.table_alias = self.item_alias()
        while self.ast.ok():
            var kind = self.join_kind()
            if kind == "":
                return
            self.join_clause(kind^)

    def join_kind(mut self) -> String:
        """The kind of the join at the cursor, or "" if there is not one."""
        if self.take_word("JOIN"):
            return String("inner")
        if self.take_word("INNER"):
            return String("inner") if self.expect_word("JOIN") else String("")
        var kind: String
        if self.at_word("LEFT"):
            kind = String("left")
        elif self.at_word("RIGHT"):
            kind = String("right")
        elif self.at_word("FULL"):
            kind = String("full")
        else:
            return String("")
        self.advance()
        _ = self.take_word("OUTER")
        return kind^ if self.expect_word("JOIN") else String("")

    def join_clause(mut self, var kind: String):
        var table = self.identifier()
        if not self.ast.ok():
            return
        var alias_name = self.item_alias()
        if not self.expect_word("ON"):
            return
        var condition = self.expression()
        if not self.ast.ok():
            return
        var left_keys = List[String]()
        var right_keys = List[String]()
        self.equijoin_keys(condition, left_keys, right_keys)
        if not self.ast.ok():
            return
        self.ast.select.joins.append(
            Join(kind^, table^, alias_name^, left_keys^, right_keys^)
        )

    def equijoin_keys(
        mut self,
        node_index: Int,
        mut left_keys: List[String],
        mut right_keys: List[String],
    ):
        """Split an `ON` condition into `a = b` conjuncts of column reads.

        Anything else records the error here, where the offending operator is
        still in hand, rather than leaving the planner to describe a tree.
        """
        # Read out before recursing: the recursive call mutates `self.ast`,
        # which invalidates any interior reference into `nodes`.
        var kind = self.ast.nodes[node_index].kind.copy()
        var a = self.ast.nodes[node_index].a
        var b = self.ast.nodes[node_index].b
        if kind == "and":
            self.equijoin_keys(a, left_keys, right_keys)
            self.equijoin_keys(b, left_keys, right_keys)
            return
        if kind != "=":
            self.fail("ON supports only equality conjuncts, found " + kind)
            return
        ref lhs = self.ast.nodes[a]
        ref rhs = self.ast.nodes[b]
        if lhs.kind != "column" or rhs.kind != "column":
            self.fail("ON supports only column = column")
            return
        left_keys.append(lhs.qualified_name())
        right_keys.append(rhs.qualified_name())

    def order_by(mut self):
        while True:
            var key = self.expression()
            if not self.ast.ok():
                return
            var ascending = not self.take_word("DESC")
            if ascending:
                _ = self.take_word("ASC")
            # SQL leaves the default engine-defined; DuckDB — which produces
            # the golden expectations — puts NULLs last in *both* directions,
            # verified against 1.5.1. Deriving it from the direction instead
            # would both answer wrongly for DESC and manufacture a conflict
            # between keys that `Sort`'s single flag cannot hold.
            var nulls_first = False
            var explicit = False
            if self.take_word("NULLS"):
                explicit = True
                if self.take_word("FIRST"):
                    nulls_first = True
                elif self.take_word("LAST"):
                    nulls_first = False
                else:
                    self.fail("expected FIRST or LAST after NULLS")
                    return
            self.ast.select.order.append(
                OrderKey(key, ascending, nulls_first, explicit)
            )
            if not self.take_punct(","):
                return


@fieldwise_init
struct Source(Copyable, Movable):
    """One named table the query may read."""

    var name: String
    var batch: RecordBatch


struct Catalog(Copyable, Movable):
    """The tables a query may name.

    A list of `Source` rather than parallel name/batch lists, and not a `Dict`
    because the binding marshals this from a Python mapping and a `Dict` would
    put an iteration order between the two sides to keep in step. A query's
    catalogue is never big enough for the linear scan to matter.

    Lookup is case-insensitive, which is SQL's rule for unquoted names.
    """

    var sources: List[Source]

    def __init__(out self):
        self.sources = List[Source]()

    def add(mut self, var name: String, var batch: RecordBatch):
        self.sources.append(Source(name^, batch^))

    def get(self, name: String) raises -> RecordBatch:
        var wanted = Ascii.upper(name)
        for ref source in self.sources:
            if Ascii.upper(source.name) == wanted:
                return source.batch.copy()
        raise Error("sql: unknown table '", name, "'")


@fieldwise_init
struct Binding(Copyable, Movable):
    """One column a reference may name, and what it resolves to."""

    var qualifier: String
    """The table alias a reference may carry."""
    var name: String
    """The column's SQL name."""
    var output: String
    """The name it actually has in the relation being built. Differs from
    `name` only after a join, where both sides are canonicalised to
    `qualifier.column` so that `emp.did` and `dept.did` stop colliding."""

    def matches(self, qualifier: String, name: String, fold: Bool) -> Bool:
        """Whether this binding answers to that reference.

        `fold` compares case-insensitively, which is SQL's rule for unquoted
        identifiers and the one `Catalog.get` applies to table names. It is a
        *second* pass rather than the only one so an exact match always wins:
        a schema holding both `id` and `ID` stays unambiguous for anyone
        spelling either exactly.
        """
        var lhs = Ascii.upper(self.name) if fold else self.name.copy()
        var rhs = Ascii.upper(name) if fold else name.copy()
        if lhs != rhs:
            return False
        if qualifier == "":
            return True
        var own = Ascii.upper(self.qualifier) if fold else self.qualifier.copy()
        return own == (Ascii.upper(qualifier) if fold else qualifier.copy())


struct Scope(Copyable, Movable):
    """What a column reference may name."""

    var bindings: List[Binding]

    def __init__(out self):
        self.bindings = List[Binding]()

    def add(
        mut self, var qualifier: String, var name: String, var output: String
    ):
        self.bindings.append(Binding(qualifier^, name^, output^))

    def count_matching(
        self, qualifier: String, name: String, fold: Bool
    ) -> Int:
        var found = 0
        for ref binding in self.bindings:
            if binding.matches(qualifier, name, fold):
                found += 1
        return found

    def has(self, qualifier: String, name: String) -> Bool:
        """Whether this scope holds such a column.

        Non-raising, unlike `resolve`, because deciding which side of a join a
        key belongs to is a question rather than a failure — `ON d.did =
        e.did` names the right table first, and that has to be answerable
        without raising on the left one.
        """
        if self.count_matching(qualifier, name, fold=False) > 0:
            return True
        return self.count_matching(qualifier, name, fold=True) > 0

    def resolve(self, qualifier: String, name: String) raises -> String:
        var fold = self.count_matching(qualifier, name, fold=False) == 0
        var matches = self.count_matching(qualifier, name, fold)
        if matches == 0:
            raise Error(
                "sql: unknown column '",
                name if qualifier == "" else qualifier + "." + name,
                "'",
            )
        if matches > 1:
            raise Error("sql: ambiguous column '", name, "'")
        for ref binding in self.bindings:
            if binding.matches(qualifier, name, fold):
                return binding.output.copy()
        raise Error("sql: unknown column '", name, "'")

    def duplicated(self, i: Int) -> Bool:
        """Whether another binding carries the same column name — the test
        that decides whether `SELECT *` may drop a qualifier."""
        for j in range(len(self.bindings)):
            if j != i and self.bindings[j].name == self.bindings[i].name:
                return True
        return False

    @staticmethod
    def split_qualified(reference: String) -> Tuple[String, String]:
        """`"e.did"` as `("e", "did")`, `"did"` as `("", "did")`."""
        for i in range(reference.byte_length()):
            if reference[byte=i] == ".":
                var qualifier = String("")
                for j in range(i):
                    qualifier += String(reference[byte=j])
                var name = String("")
                for j in range(i + 1, reference.byte_length()):
                    name += String(reference[byte=j])
                return (qualifier^, name^)
        return (String(""), reference.copy())

    @staticmethod
    def position(names: List[String], name: String) -> Int:
        """Where `name` sits in `names`, or -1."""
        for i in range(len(names)):
            if names[i] == name:
                return i
        return -1


struct Planner(Copyable, Movable):
    var ast: Ast
    var catalog: Catalog
    var scope: Scope

    var agg_nodes: List[Int]
    """AST index of each aggregate call found in the query."""
    var agg_values: List[DynValue]
    var agg_outputs: List[String]
    """`__agg0`, `__agg1`, ... — generated so two aggregates with the same SQL
    spelling cannot collide in the grouped relation's schema."""

    var key_outputs: List[String]
    """The name each `GROUP BY` key has in the grouped relation."""
    var key_nodes: List[Int]
    """The AST index of each key, for matching a SELECT item against it."""

    var grouped: Bool
    """Whether expressions are now read against the grouped relation. Set once
    the `GROUP BY` keys have been translated — they are the last thing
    expressed in the source's terms."""

    var schema: Schema
    """The schema an expression is currently evaluated against — the source's
    before grouping, the grouped relation's after. Carried because a few
    operators cannot be chosen without knowing an operand's dtype: marrow's
    `ne` dispatches on primitives and booleans are not primitive, so SQL's
    `p <> q` over two `bool` columns has to become `xor`."""

    var order_keys: List[String]
    """The output column each `ORDER BY` key sorts on."""
    var order_extras: List[String]
    var order_extra_values: List[DynValue]
    """Columns added to the projection only so a sort key exists, and dropped
    again once the sort has run."""

    def __init__(out self, var ast: Ast, var catalog: Catalog):
        self.ast = ast^
        self.catalog = catalog^
        self.scope = Scope()
        self.agg_nodes = List[Int]()
        self.agg_values = List[DynValue]()
        self.agg_outputs = List[String]()
        self.key_outputs = List[String]()
        self.key_nodes = List[Int]()
        self.grouped = False
        self.schema = Schema()
        self.order_keys = List[String]()
        self.order_extras = List[String]()
        self.order_extra_values = List[DynValue]()

    # -- expressions --------------------------------------------------------

    def translate(mut self, index: Int) raises -> RuntimeValue:
        """One expression, evaluated against the relation currently in hand.

        **One walk, not two.** After grouping, an expression reads differently
        only at its *substitution points*: an aggregate call becomes a read of
        the column the grouping produced, and a subtree equal to a group key
        becomes a read of that key. Everything below those points is the same
        tree it always was, so `grouped` is answered here and the rest of the
        method is shared. A second traversal for the grouped case is what this
        replaces, and it handled five node kinds of fifteen — `CASE`, `IN` and
        `BETWEEN` over an aggregate all failed.

        The dispatcher. A node whose tag needs more than a line gets a
        `visit_<tag>` method, so the tag names the method that handles it and
        nothing has to be looked up to find out which; the rest are answered
        inline. Mojo has no dynamic dispatch, so the chain below *is* the
        vtable — the naming is what keeps it readable.
        """
        if index == NONE:
            raise Error("sql: missing expression")
        if self.grouped:
            if self.is_aggregate_call(index):
                return column(self.collect_aggregate(index))
            # A group key, recognised **structurally**: `SELECT (v > 3) ...
            # GROUP BY (v > 3)` is two AST nodes spelling one expression, and
            # after grouping neither `v` nor `3` exists any more — only the
            # key column does. Comparing the names the two expressions give
            # themselves cannot see that, because a computed key names itself
            # "".
            for i in range(len(self.key_nodes)):
                if self.same_expr(index, self.key_nodes[i]):
                    return column(self.key_outputs[i].copy())
        # A copy, not a borrow: translating a child may register an
        # aggregate, which mutates `self.ast`'s owner and would invalidate an
        # interior reference.
        var node = self.ast.nodes[index].copy()
        if node.kind == "column":
            return column(self.scope.resolve(node.qualifier, node.text))
        if node.kind == "int":
            return literal(self.integer_literal(Int(node.text)))
        if node.kind == "float":
            return literal(Float64Scalar(Float64(node.text)).to_dyn())
        if node.kind == "string":
            return literal(StringScalar(node.text.copy()).to_dyn())
        if node.kind == "bool":
            return literal(BoolScalar(node.value == 1).to_dyn())
        if node.kind == "null":
            return literal(NullScalar().to_dyn())
        if node.kind == "neg":
            return neg(self.translate(node.a))
        if node.kind == "not":
            return not_(self.translate(node.a))
        if node.kind == "is_null":
            return is_null(self.translate(node.a))
        if node.kind == "is_not_null":
            return is_valid(self.translate(node.a))
        if node.kind == "cast":
            return cast(self.translate(node.a), Planner.dtype_for(node.text))
        if node.kind == "star":
            raise Error("sql: '*' is only allowed in the SELECT list")
        if node.kind == "case":
            return self.visit_case(node)
        if node.kind == "func":
            return self.visit_func(node)
        if node.kind == "in" or node.kind == "not_in":
            return self.visit_in(node)
        if node.kind.endswith("like"):
            return self.visit_like(node)
        if node.kind.endswith("between"):
            # `lo <= x <= hi`, translated twice rather than shared: the value
            # is an expression, and marrow has no let-binding to hang it on.
            var lower_bound = ge(self.translate(node.a), self.translate(node.b))
            var upper_bound = le(self.translate(node.a), self.translate(node.c))
            var between = and_(lower_bound^, upper_bound^)
            return not_(between^) if node.kind == "not_between" else between^
        # Anything left is a binary operator, and its tag *is* the operator.
        return self.apply_operator(
            node.kind, self.translate(node.a), self.translate(node.b)
        )

    @staticmethod
    def is_aggregate_name(name: String) -> Bool:
        """Whether a function name makes the call an aggregate. `AVG` and
        `MEAN` are the same function under SQL's name and marrow's."""
        return Ascii.contains_word(
            (
                "SUM COUNT AVG MEAN MIN MAX PRODUCT STDDEV STDDEV_POP"
                " STDDEV_SAMP VARIANCE VAR_POP VAR_SAMP"
            ),
            name,
        )

    @staticmethod
    def dtype_for(sql_type: String) raises -> DynType:
        """A SQL type name as marrow's `DataType`.

        Widths follow DuckDB, which produces the golden expectations: `INTEGER` is
        32 bits and `BIGINT` is 64, so a corpus that casts to `INTEGER` and
        compares against DuckDB's answer needs the same choice here.
        """
        var name = sql_type
        if name == "TINYINT":
            return DynType(int8)
        if name == "SMALLINT" or name == "INT2":
            return DynType(int16)
        if name == "INT" or name == "INTEGER" or name == "INT4":
            return DynType(int32)
        if name == "BIGINT" or name == "INT8":
            return DynType(int64)
        if name == "UTINYINT":
            return DynType(uint8)
        if name == "USMALLINT":
            return DynType(uint16)
        if name == "UINTEGER":
            return DynType(uint32)
        if name == "UBIGINT":
            return DynType(uint64)
        if name == "REAL" or name == "FLOAT4" or name == "FLOAT":
            # `FLOAT` is an alias for `REAL` in DuckDB — 32 bits, not 64.
            return DynType(float32)
        if name == "DOUBLE" or name == "DOUBLE PRECISION" or name == "FLOAT8":
            return DynType(float64)
        if (
            name == "VARCHAR"
            or name == "TEXT"
            or name == "STRING"
            or name == "CHAR"
        ):
            return DynType(string)
        if name == "BOOLEAN" or name == "BOOL":
            return DynType(bool_)
        if name == "DATE":
            return DynType(date32())
        if name == "TIMESTAMP":
            return DynType(timestamp(microsecond))
        raise Error("sql: unsupported type '", name, "'")

    def integer_literal(self, value: Int) raises -> DynScalar:
        """An integer constant, typed by magnitude the way DuckDB types one.

        `qty * 2` over an `int32` column must stay `int32`, and it only does
        if the literal is `int32` too — marrow's promotion widens, so an
        `int64` literal would drag the whole expression up a width that SQL
        does not.

        Typing by **magnitude** rather than by the other operand is what keeps
        this safe: `promote_dyn`'s docstring records that narrowing a literal
        to match a column made `int32_col > 2**40` raise instead of compare.
        A literal that does not fit in `int32` simply is `int64`, and widening
        takes it from there.
        """
        var fits_in_int32 = value >= -2147483648 and value <= 2147483647
        if fits_in_int32:
            return Int32Scalar(Int32(value)).to_dyn()
        return Int64Scalar(Int64(value)).to_dyn()

    def both_boolean(self, l: RuntimeValue, r: RuntimeValue) -> Bool:
        """Whether both operands are `bool` against the current schema.

        Non-raising: an operand whose dtype cannot be resolved yet is simply
        not known to be boolean, and the ordinary kernels give the better
        error message for whatever is actually wrong with it.
        """
        try:
            return (
                l.dtype(self.schema).is_bool()
                and r.dtype(self.schema).is_bool()
            )
        except:
            return False

    def visit_like(mut self, node: Node) raises -> RuntimeValue:
        var pattern = self.ast.nodes[node.b].copy()
        if pattern.kind != "string":
            raise Error("sql: LIKE needs a literal pattern")
        var value = self.translate(node.a)
        var matched = ilike(value^, pattern.text.copy()) if node.kind.endswith(
            "ilike"
        ) else like(value^, pattern.text.copy())
        return not_(matched^) if node.kind.startswith("not_") else matched^

    def visit_in(mut self, node: Node) raises -> RuntimeValue:
        """`x IN (...)` as an `OR` chain, with NULLs lifted out of it.

        `x = NULL` is NULL for every row, so a NULL in the list cannot join
        the chain as an equality. Its whole effect is to turn the chain's *false* into NULL,
        which is SQL's rule and what makes `NOT IN (1, NULL)` match nothing:

            IN  with a NULL present -> TRUE where the chain matches, else NULL
            NOT IN                  -> FALSE where it matches, else NULL
        """
        if node.kids_len == 0:
            raise Error("sql: IN needs at least one value")
        var any_match = Optional[RuntimeValue](None)
        var has_null = False
        for i in range(node.kids_len):
            var item = self.ast.kid(node, i)
            if self.ast.nodes[item].kind == "null":
                has_null = True
                continue
            var same = eq(self.translate(node.a), self.translate(item))
            any_match = Optional(
                or_(any_match.value().copy(), same^)
            ) if any_match else Optional(same^)
        if not any_match:
            # `x IN (NULL)` — nothing to compare against, so always NULL.
            any_match = Optional(literal(BoolScalar(False).to_dyn()))
        var matched = any_match.value().copy()
        if has_null:
            var conditions = List[RuntimeValue]()
            var results = List[RuntimeValue]()
            conditions.append(matched^)
            results.append(literal(BoolScalar(True).to_dyn()))
            matched = case_when(
                conditions^, results^, Optional[RuntimeValue](None)
            )
        return not_(matched^) if node.kind == "not_in" else matched^

    def visit_case(mut self, node: Node) raises -> RuntimeValue:
        """`kids` alternates condition and result, so the two lists
        `case_when` wants are the even and odd positions."""
        var conditions = List[RuntimeValue]()
        var results = List[RuntimeValue]()
        for i in range(node.kids_len):
            var arm = self.translate(self.ast.kid(node, i))
            if i % 2 == 0:
                conditions.append(arm^)
            else:
                results.append(arm^)
        # A missing ELSE is NULL, which is SQL's rule and `case_when`'s too.
        var otherwise = Optional[RuntimeValue](
            None
        ) if node.a == NONE else Optional(self.translate(node.a))
        return case_when(conditions^, results^, otherwise^)

    def visit_func(mut self, node: Node) raises -> RuntimeValue:
        """A scalar function call. Aggregates never reach here — `rewrite`
        replaces them with a column read before translation."""
        if node.text == "DATE_TRUNC" and node.kids_len == 2:
            # The unit is a literal the kernel wants as text rather than as a
            # column, so it never becomes a child.
            var unit = self.ast.nodes[self.ast.kid(node, 0)].copy()
            if unit.kind != "string":
                raise Error("sql: DATE_TRUNC needs a literal unit")
            return date_trunc(
                self.translate(self.ast.kid(node, 1)), unit.text.copy()
            )
        var args = List[RuntimeValue]()
        for i in range(node.kids_len):
            args.append(self.translate(self.ast.kid(node, i)))
        return self.apply_function(node.text, args^)

    def apply_function(
        mut self, name: String, var args: List[RuntimeValue]
    ) raises -> RuntimeValue:
        """A scalar function over operands that are already translated.

        Three shapes need code, and each is a case where SQL's meaning is not
        one marrow node. Everything else — sixty-odd names — is a **rename**
        onto marrow's tag, so it lives in `function_tag` as data and shares
        this one construction.
        """
        if name == "COALESCE":
            return coalesce(args^)
        if name == "GREATEST" or name == "LEAST":
            if len(args) != 2:
                raise Error("sql: ", name, " takes two arguments")
            # SQL's extrema **skip** nulls — a row with one null operand
            # answers with the other — while `maximum`/`minimum` propagate
            # them. `coalesce` over the pair recovers exactly SQL's rule.
            var extremum = maximum(
                args[0].copy(), args[1].copy()
            ) if name == "GREATEST" else minimum(args[0].copy(), args[1].copy())
            var fallbacks = List[RuntimeValue]()
            fallbacks.append(extremum^)
            fallbacks.append(args[0].copy())
            fallbacks.append(args[1].copy())
            return coalesce(fallbacks^)
        if name == "ISODOW":
            if len(args) != 1:
                raise Error("sql: ISODOW takes one argument")
            # marrow counts from Monday=0, ISO 8601 from Monday=1. DuckDB's
            # `dayofweek` is a third convention (Sunday=0) and stays unmapped
            # rather than guessed at.
            return add(
                day_of_week(args[0].copy()),
                literal(Int32Scalar(1).to_dyn()),
            )
        var tag = Planner.tag_for(name, len(args))
        if tag == "":
            raise Error(
                "sql: unsupported function '",
                name,
                "' of ",
                String(len(args)),
                " argument(s)",
            )
        return RuntimeValue(tag^, args^)

    @staticmethod
    def tag_for(name: String, arity: Int) -> String:
        """The marrow tag for a SQL function name, or "".

        A table, because every entry is a pure rename and the code that would
        surround one — build the operands, call the builder, return — is the
        same for all of them and is written once in `invoke`. Adding a
        function is a word here.

        Split by arity because SQL overloads on it: `TRIM(s)` strips
        whitespace while `TRIM(s, chars)` strips a set.
        """
        if arity == 1:
            return Ascii.lookup(
                (
                    "ABS:abs CEIL:ceil CEILING:ceil FLOOR:floor ROUND:round"
                    " TRUNC:trunc SIGN:sign SQRT:sqrt EXP:exp EXP2:exp2 LN:ln"
                    " LOG:log10 LOG10:log10 LOG2:log2 LOG1P:log1p SIN:sin"
                    " COS:cos ISNAN:is_nan ISINF:is_inf UPPER:upper UCASE:upper"
                    " LOWER:lower LCASE:lower TRIM:strip BTRIM:strip"
                    " LTRIM:lstrip RTRIM:rstrip REVERSE:reverse"
                    " LENGTH:char_length CHAR_LENGTH:char_length"
                    " CHARACTER_LENGTH:char_length STRLEN:length"
                    " OCTET_LENGTH:length ASCII:ascii ARRAY_LENGTH:array_length"
                    " CARDINALITY:array_length LEN:array_length YEAR:year"
                    " MONTH:month DAY:day HOUR:hour MINUTE:minute SECOND:second"
                    " QUARTER:quarter WEEK:week ISOYEAR:iso_year EPOCH:epoch"
                    " LAST_DAY:last_day DAYNAME:day_name MONTHNAME:month_name"
                    " DAYOFYEAR:day_of_year"
                ),
                name,
            )
        if arity == 2:
            return Ascii.lookup(
                (
                    "POW:pow POWER:pow MOD:mod NULLIF:nullif IFNULL:fill_null"
                    " STARTSWITH:startswith STARTS_WITH:startswith"
                    " ENDSWITH:endswith ENDS_WITH:endswith CONTAINS:contains"
                    " POSITION:position STRPOS:position INSTR:position"
                    " LEFT:left RIGHT:right REPEAT:repeat TRIM:trim_chars"
                    " BTRIM:trim_chars"
                ),
                name,
            )
        if arity == 3:
            return Ascii.lookup(
                (
                    "SUBSTR:substr SUBSTRING:substr REPLACE:replace"
                    " SPLIT_PART:split_part LPAD:lpad RPAD:rpad"
                ),
                name,
            )
        return String("")

    def is_aggregate_call(self, index: Int) -> Bool:
        if index == NONE:
            return False
        ref node = self.ast.nodes[index]
        return node.kind == "func" and Planner.is_aggregate_name(node.text)

    def contains_aggregate(self, index: Int) -> Bool:
        """Whether this subtree has an aggregate anywhere in it — what decides
        that a query is grouped even without a `GROUP BY`."""
        if index == NONE:
            return False
        if self.is_aggregate_call(index):
            return True
        ref node = self.ast.nodes[index]
        var kids = [node.a, node.b, node.c]
        for kid in kids:
            if self.contains_aggregate(kid):
                return True
        for i in range(node.kids_len):
            if self.contains_aggregate(self.ast.kid(node, i)):
                return True
        return False

    def collect_aggregate(mut self, index: Int) raises -> String:
        """Register the aggregate call at `index`, and answer the name its
        result will have in the grouped relation."""
        for i in range(len(self.agg_nodes)):
            if self.agg_nodes[i] == index:
                return self.agg_outputs[i].copy()
        var node = self.ast.nodes[index].copy()
        var output = "__agg" + String(len(self.agg_nodes))
        var value = self.to_aggregate(node, output)
        self.agg_nodes.append(index)
        self.agg_values.append(DynValue(value^))
        self.agg_outputs.append(output.copy())
        return output^

    def to_aggregate(
        mut self, node: Node, var output: String
    ) raises -> RuntimeAggregate:
        if node.kids_len != 1:
            raise Error("sql: ", node.text, " takes exactly one argument")
        var name = node.text
        var arg_kind = self.ast.nodes[self.ast.kid(node, 0)].kind.copy()
        var input: RuntimeValue
        if arg_kind == "star":
            # `COUNT(*)` counts rows; `count` counts non-null values, so the
            # input is a constant that is never null.
            if name != "COUNT":
                raise Error("sql: '*' is only an argument to COUNT")
            input = literal(Int64Scalar(Int64(1)).to_dyn())
        else:
            input = self.translate(self.ast.kid(node, 0))
        var distinct = node.distinct
        if name == "SUM":
            return input^.sum().alias(output^)
        if name == "COUNT":
            var counted = (
                input^.count_distinct() if distinct else input^.count()
            )
            return counted^.alias(output^)
        if name == "AVG" or name == "MEAN":
            return input^.mean().alias(output^)
        if name == "MIN":
            return input^.min().alias(output^)
        if name == "MAX":
            return input^.max().alias(output^)
        if name == "PRODUCT":
            return input^.product().alias(output^)
        if name == "STDDEV" or name == "STDDEV_POP":
            return input^.stddev().alias(output^)
        if name == "STDDEV_SAMP":
            return input^.stddev_samp().alias(output^)
        if name == "VARIANCE" or name == "VAR_POP":
            return input^.variance().alias(output^)
        if name == "VAR_SAMP":
            return input^.var_samp().alias(output^)
        raise Error("sql: unsupported aggregate '", name, "'")

    def same_expr(self, a: Int, b: Int) -> Bool:
        """Whether two AST nodes spell the same expression.

        A structural walk rather than a rendered-string comparison: the flat
        tree makes it a field-by-field test, and nothing has to agree on how
        an expression prints.
        """
        if a == b:
            return True
        if a == NONE or b == NONE:
            return False
        ref x = self.ast.nodes[a]
        ref y = self.ast.nodes[b]
        var shallow = (
            x.kind == y.kind
            and x.text == y.text
            and x.qualifier == y.qualifier
            and x.value == y.value
            and x.distinct == y.distinct
            and x.kids_len == y.kids_len
        )
        if not shallow:
            return False
        if (
            not self.same_expr(x.a, y.a)
            or not self.same_expr(x.b, y.b)
            or not self.same_expr(x.c, y.c)
        ):
            return False
        for i in range(x.kids_len):
            var left = self.ast.kids[x.kids_start + i]
            var right = self.ast.kids[y.kids_start + i]
            if not self.same_expr(left, right):
                return False
        return True

    def apply_operator(
        mut self, op: String, var lhs: RuntimeValue, var rhs: RuntimeValue
    ) raises -> RuntimeValue:
        """A binary operator over two values that are already translated.

        Reached from both phases: `translate` calls it with the operands as
        written, `rewrite` with the operands as they read after grouping.

        Three operators need code; the rest are a rename onto marrow's own
        tag, which `RuntimeValue` takes directly.
        """
        if op == "=" or op == "<>":
            # `ne`/`eq` dispatch on primitives, and `bool` is bit-packed
            # rather than primitive, so comparing two boolean columns reaches
            # `dispatch_primitive` and aborts. `xor` is the same question
            # asked in the form marrow answers.
            if self.both_boolean(lhs, rhs):
                var differ = xor(lhs^, rhs^)
                return differ^ if op == "<>" else not_(differ^)
            return eq(lhs^, rhs^) if op == "=" else ne(lhs^, rhs^)
        if op == "and":
            # `and_`/`or_` fold constants at construction, which is what lets
            # `PropagateEmpty` collapse a subtree; the raw node would not.
            return and_(lhs^, rhs^)
        if op == "or":
            return or_(lhs^, rhs^)
        var tag = Ascii.lookup(
            "+:add -:sub *:mul /:truediv %:mod <:lt <=:le >:gt >=:ge", op
        )
        if tag == "":
            raise Error("sql: unsupported operator '", op, "'")
        return RuntimeValue(tag^, lhs, rhs)

    def source(mut self) raises -> DynRelation:
        """`FROM` and its joins, with the scope they establish."""
        ref select = self.ast.select
        if select.table == "":
            raise Error("sql: FROM is required")
        var batch = self.catalog.get(select.table)
        var relation = table(batch^)
        var qualifier = (
            select.table_alias.copy() if select.table_alias
            != "" else select.table.copy()
        )
        for ref name in relation.schema().names():
            self.scope.add(qualifier.copy(), name.copy(), name.copy())
        if len(select.joins) == 0:
            return relation^
        return self.joins(relation^)

    def joins(mut self, var relation: DynRelation) raises -> DynRelation:
        """Each join, after canonicalising both sides to `qualifier.column`.

        marrow's `Join` concatenates the two schemas positionally, so `emp.did`
        and `dept.did` would both arrive as `did` and every later reference to
        either would be ambiguous. Renaming first costs one `Project` per join
        and makes the ambiguity impossible instead of merely unlikely.
        """
        relation = self.canonicalise(relation^)
        for ref join in self.ast.select.joins:
            var right_batch = self.catalog.get(join.table)
            var right = table(right_batch^)
            var right_qualifier = (
                join.alias_name.copy() if join.alias_name
                != "" else join.table.copy()
            )
            var right_scope = Scope()
            for ref name in right.schema().names():
                right_scope.add(
                    right_qualifier.copy(),
                    name.copy(),
                    right_qualifier + "." + name,
                )
            var right_names = List[String](capacity=len(right_scope.bindings))
            for ref binding in right_scope.bindings:
                right_names.append(binding.output.copy())
            right = Planner.rename(right^, right_names)

            var left_keys = List[Int]()
            var right_keys = List[Int]()
            var left_schema = relation.schema()
            var right_schema = right.schema()
            for i in range(len(join.left_keys)):
                var a = Scope.split_qualified(join.left_keys[i])
                var b = Scope.split_qualified(join.right_keys[i])
                # Either side of the `ON` may be written first, so the side a
                # key belongs to is decided by which relation has the column.
                # `has` and not `resolve`: `ON d.did = e.did` asks the left
                # scope about a right-hand column, which is a question rather
                # than an error.
                var a_on_left = self.scope.has(a[0], a[1])
                var left_ref = a if a_on_left else b
                var right_ref = b if a_on_left else a
                left_keys.append(
                    left_schema.get_field_index(
                        self.scope.resolve(left_ref[0], left_ref[1])
                    )
                )
                right_keys.append(
                    right_schema.get_field_index(
                        right_scope.resolve(right_ref[0], right_ref[1])
                    )
                )
            for i in range(len(left_keys)):
                if left_keys[i] < 0 or right_keys[i] < 0:
                    raise Error("sql: join key not found in either table")
            relation = relation.join(
                right^, left_keys^, right_keys^, join.to_kind()
            )
            for i in range(len(right_scope.bindings)):
                self.scope.add(
                    right_scope.bindings[i].qualifier.copy(),
                    right_scope.bindings[i].name.copy(),
                    right_scope.bindings[i].output.copy(),
                )
        return relation^

    @staticmethod
    def rename(
        var relation: DynRelation, names: List[String]
    ) raises -> DynRelation:
        var values = List[DynValue]()
        for ref name in relation.schema().names():
            values.append(DynValue(column(name.copy())))
        return relation.project(names.copy(), values^)

    def canonicalise(mut self, var relation: DynRelation) raises -> DynRelation:
        """Rename every column of the left-hand side to `qualifier.column`."""
        var renamed = List[String]()
        for i in range(len(self.scope.bindings)):
            var canonical = (
                self.scope.bindings[i].qualifier
                + "."
                + self.scope.bindings[i].name
            )
            renamed.append(canonical.copy())
            self.scope.bindings[i].output = canonical^
        return Planner.rename(relation^, renamed)

    def collect_projection(
        mut self,
    ) raises -> Tuple[List[String], List[DynValue]]:
        """The `SELECT` list as output names and values.

        `*` expands to every column in scope, which is also where a joined
        query sheds the `qualifier.column` names it carried internally.
        """
        var names = List[String]()
        var values = List[DynValue]()
        var select = self.ast.select.copy()
        for i in range(len(select.items)):
            var index = select.items[i].node
            var node = self.ast.nodes[index].copy()
            if node.kind == "star":
                for j in range(len(self.scope.bindings)):
                    if (
                        node.qualifier != ""
                        and self.scope.bindings[j].qualifier != node.qualifier
                    ):
                        continue
                    # After a join both sides may carry the same column name,
                    # and two output columns called `did` are indistinguishable
                    # to every consumer — `to_pylist()` silently keeps one. The
                    # qualifier is kept exactly where it is needed to tell them
                    # apart, and dropped everywhere else.
                    var bare = self.scope.bindings[j].name.copy()
                    names.append(
                        self.scope.bindings[j].qualifier
                        + "."
                        + bare if self.scope.duplicated(j) else bare.copy()
                    )
                    values.append(
                        DynValue(column(self.scope.bindings[j].output.copy()))
                    )
                continue
            var value = self.translate(index)
            ref alias_name = select.items[i].alias_name
            var name = (
                alias_name.copy() if alias_name
                != "" else self.item_name(index, value)
            )
            names.append(name^)
            values.append(DynValue(value^))
        return (names^, values^)

    def item_name(self, index: Int, value: RuntimeValue) raises -> String:
        """The name an unaliased item takes: the column's own name where it is
        a bare column read, and otherwise whatever the expression calls
        itself."""
        # A copy, not a borrow: translating a child may register an
        # aggregate, which mutates `self.ast`'s owner and would invalidate an
        # interior reference.
        var node = self.ast.nodes[index].copy()
        if node.kind == "column":
            return node.text.copy()
        return value.name()

    def is_grouped(self) -> Bool:
        if len(self.ast.select.group_by) > 0:
            return True
        for ref item in self.ast.select.items:
            if self.contains_aggregate(item.node):
                return True
        return self.contains_aggregate(self.ast.select.having)

    def collect_group_keys(mut self) raises -> List[DynValue]:
        """The `GROUP BY` keys, and the name each will have once grouped.

        Two rules, neither of them free:

        - An ordinal groups by the *SELECT item* at that position, as
          `ORDER BY 1` sorts by one. Without it `GROUP BY 1` groups every row
          into the single group "the literal 1".
        - A computed key has no name of its own, and `Aggregate._output_schema`
          calls it `key0`, `key1`, … by position. Recording `key.name()` — an
          empty string — instead is what made `GROUP BY UPPER(k)` fail with
          `column '' not found in schema`.
        """
        ref select = self.ast.select
        var keys = List[DynValue]()
        for i in range(len(select.group_by)):
            var index = select.group_by[i]
            ref node = self.ast.nodes[index]
            if node.kind == "int":
                var position = node.value - 1
                if position < 0 or position >= len(select.items):
                    raise Error("sql: GROUP BY ordinal out of range")
                index = select.items[position].node
            var key = self.translate(index)
            var name = key.name()
            self.key_outputs.append(name if name != "" else "key" + String(i))
            self.key_nodes.append(index)
            keys.append(DynValue(key^))
        return keys^

    def build(mut self) raises -> DynRelation:
        var relation = self.source()
        self.schema = relation.schema()
        ref select = self.ast.select

        if select.predicate != NONE:
            if self.contains_aggregate(select.predicate):
                raise Error("sql: WHERE cannot contain an aggregate")
            relation = relation.filter(self.translate(select.predicate))

        var projected: Tuple[List[String], List[DynValue]]
        var having = Optional[RuntimeValue](None)
        if self.is_grouped():
            # The keys are the last thing expressed in the source's terms;
            # everything after this reads against the grouped relation.
            var keys = self.collect_group_keys()
            self.grouped = True
            # Everything that can name an aggregate must be visited *before*
            # the node is built, because visiting is what registers them and
            # `Aggregate` takes the whole list up front. `ORDER BY SUM(v)` is
            # the one that is easy to forget: it is applied last but resolved
            # here.
            projected = self.collect_projection()
            if select.having != NONE:
                having = Optional(self.translate(select.having))
            self.collect_order_keys(projected[0])
            relation = relation.aggregate(self.agg_values.copy(), keys^)
            if having:
                relation = relation.filter(having.value().copy())
        else:
            if select.having != NONE:
                raise Error("sql: HAVING needs GROUP BY or an aggregate")
            projected = self.collect_projection()
            self.collect_order_keys(projected[0])

        var names = projected[0].copy()
        var values = projected[1].copy()
        for i in range(len(self.order_extras)):
            names.append(self.order_extras[i].copy())
            values.append(self.order_extra_values[i].copy())
        relation = relation.project(names^, values^)

        if select.distinct:
            if len(self.order_extras) > 0:
                raise Error(
                    "sql: SELECT DISTINCT with an ORDER BY key that is not a"
                    " selected column is not supported"
                )
            # No `Distinct` node exists; grouping by every output column is
            # the same relation.
            var keys = List[DynValue]()
            for ref name in relation.schema().names():
                keys.append(DynValue(column(name.copy())))
            relation = relation.aggregate(List[DynValue](), keys^)

        if len(select.order) > 0:
            relation = self.apply_order(relation^)
        if len(self.order_extras) > 0:
            # The extras existed only to be sorted on.
            relation = relation.select(projected[0].copy())

        if select.limit >= 0 or select.offset > 0:
            var length = select.limit if select.limit >= 0 else _UNLIMITED
            relation = relation.limit(length, select.offset)
        return relation^

    def collect_order_keys(mut self, output_names: List[String]) raises:
        """Decide which column each `ORDER BY` key sorts on.

        `Sort` runs after the projection, so a key must name one of its
        columns. Three ways it can:

        - an ordinal (`ORDER BY 1`) names one by position;
        - a bare column name matches an output, which is how `ORDER BY total`
          finds the alias it was just given;
        - anything else is an expression the projection does not contain, so
          it is *added* to the projection under a generated name and dropped
          again after the sort. That is what makes `ORDER BY SUM(v)` work
          without `SUM(v)` being selected.

        Called before the `Aggregate` node is built, because the third case
        registers aggregates.
        """
        ref select = self.ast.select
        for i in range(len(select.order)):
            var index = select.order[i].node
            ref node = self.ast.nodes[index]
            if node.kind == "int":
                var position = node.value - 1
                if position < 0 or position >= len(output_names):
                    raise Error("sql: ORDER BY ordinal out of range")
                self.order_keys.append(output_names[position].copy())
                continue
            if (
                node.kind == "column"
                and Scope.position(output_names, node.text) >= 0
            ):
                self.order_keys.append(node.text.copy())
                continue
            var extra = "__ord" + String(len(self.order_extras))
            var value = self.translate(index)
            self.order_extras.append(extra.copy())
            self.order_extra_values.append(DynValue(value^))
            self.order_keys.append(extra^)

    def apply_order(mut self, var relation: DynRelation) raises -> DynRelation:
        ref select = self.ast.select
        var keys = List[DynValue]()
        for ref name in self.order_keys:
            keys.append(DynValue(column(name.copy())))
        var ascending = List[Bool](capacity=len(select.order))
        for ref key in select.order:
            ascending.append(key.ascending)
        # `Sort` carries one `nulls_first` for every key (CLAUDE.md), so two
        # keys that ask for different placements cannot both be honoured. Only
        # an **explicit** clause counts: the default is the same for every key,
        # so deriving one per key would reject `ORDER BY a ASC, b DESC` for a
        # conflict the query never asked for.
        var nulls_first = select.order[0].nulls_first
        for ref key in select.order:
            var conflicting = key.nulls_first != nulls_first
            if conflicting and key.nulls_explicit:
                raise Error(
                    "sql: per-key NULLS FIRST/LAST is not supported; Sort"
                    " carries one flag for all keys"
                )
        return relation.sort_by(keys^, ascending^, nulls_first)


comptime _UNLIMITED = 1 << 62
"""`OFFSET` without `LIMIT` still needs a length, and `Limit` takes one."""


def sql(var query: String, var catalog: Catalog) raises -> DynRelation:
    """`query` as a plan against `catalog`, or its parse error as a raise.

    The one entry point, and a verb rather than a type: everything else in
    this file is a type because it holds state across a stage, whereas this
    holds nothing — it is `table()` and `scan()`'s sibling in
    `expr/builders.mojo`, which are free functions for the same reason.

    `Parser.parse` cannot raise — that is what keeps it comptime-eligible —
    so this is where a recorded error becomes one.
    """
    var ast = Parser.parse(query^)
    if not ast.ok():
        raise Error(
            "sql: ", ast.error, " (at offset ", String(ast.error_pos), ")"
        )
    var planner = Planner(ast^, catalog^)
    return planner.build()
