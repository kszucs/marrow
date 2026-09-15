"""Parameters: what a plan declares, and the values one execution binds.

`ParamSpec` is the declaration side -- what a parameter is called, what it
holds, and how a command line spells it -- and `Bindings` is the value side.
Both are leaves of the expression package: neither imports anything from
`marrow.expr`, which is what lets `logical.mojo`, the comptime leaves and
`cli.mojo` all depend on this module without a cycle.
"""

from std.collections import Dict

from ..dtypes import DynType, NumericType, StringLikeType
from ..scalars import BinaryLikeScalar, BoolScalar, DynScalar, PrimitiveScalar
from ..utils.argparse import parse_bool


comptime Bindings = Dict[String, DynScalar]
"""Parameter values for one execution: a plain name -> scalar map.

An alias, not a struct. It only ever wrapped a `Dict[String, DynScalar]` with
a `set`/`get` pair that `Dict` already provides — and `Dict.get` does not even
raise, where the wrapper's did. Being an alias means a caller writes a dict
literal:

    plan.execute(bindings={"min-a": Int64Scalar(4).to_dyn()})

Passed to `to_operator`, not stored on the plan, which is what keeps a plan
immutable and lets two executions use different values without interfering.

Missing names are not an error here — a parameter with a default is satisfied
without one, and `NumericParam` raises naming itself when it has neither.
"""


struct ParamSpec(Copyable, Movable):
    """A parameter a plan declares: what a caller must, or may, bind.

    Holds no `DynScalar`: a `List` of a struct carrying one is the
    List-of-Variant growth defect CLAUDE.md records, and the default is only
    ever *shown* here — the node itself applies it.
    """

    var name: String
    var dtype: DynType
    var help: String
    var default: Optional[String]
    """The default as a command line would spell it; `None` when required."""

    var parse: Optional[def(String) thin raises -> DynScalar]
    """A command-line token as this parameter's scalar, or `None` when the dtype
    has no command-line spelling. Instantiated per dtype a plan names, so a
    binary links only the parsers its parameters need."""

    def __init__(
        out self,
        var name: String,
        var dtype: DynType,
        var help: String,
        var default: Optional[String],
        parse: Optional[def(String) thin raises -> DynScalar],
    ):
        self.name = name^
        self.dtype = dtype^
        self.help = help^
        self.default = default^
        self.parse = parse


def numeric_from_text[T: NumericType](text: String) raises -> DynScalar:
    """`text` as a `T` scalar, refusing a value `T` cannot hold.

    A narrowing that changes the value is an error rather than a wrap: `300`
    is not a `uint8`, and a negative number is not an unsigned one.
    """
    comptime if T.native.is_floating_point():
        var value: Float64
        try:
            value = atof(text)
        except:
            raise Error("expected ", T(), ", got '", text, "'")
        return PrimitiveScalar[T](Scalar[T.native](value)).to_dyn()
    else:
        var wide: Int
        try:
            wide = atol(text)
        except:
            raise Error("expected ", T(), ", got '", text, "'")
        var narrow = Scalar[T.native](wide)
        comptime if T.native.is_unsigned():
            if wide < 0 or Int(narrow) != wide:
                raise Error("'", text, "' is out of range for ", T())
        else:
            if Int(narrow) != wide:
                raise Error("'", text, "' is out of range for ", T())
        return PrimitiveScalar[T](narrow).to_dyn()


def bool_from_text(text: String) raises -> DynScalar:
    """`true`/`false`/`1`/`0` as a `bool` scalar — `parse_bool`'s spellings."""
    return BoolScalar(parse_bool(text)).to_dyn()


def string_from_text[T: StringLikeType](text: String) raises -> DynScalar:
    """`text` itself, as a `T` scalar."""
    return BinaryLikeScalar[T](text).to_dyn()


def distinct_params(specs: List[ParamSpec]) -> List[ParamSpec]:
    """`specs` with one entry per name, in first-seen order.

    A name read twice is one parameter — `Bindings` is keyed by name, so every
    read sees the same value — and its first declaration is the one reported.
    A later read of another dtype is refused when the value binds, naming the
    parameter, so a conflict still surfaces; comparing dtypes and defaults here
    as well measured 832 bytes of `__text` on `query_cli` for no earlier
    answer.
    """
    var out = List[ParamSpec]()
    for ref spec in specs:
        var seen = False
        for ref kept in out:
            if kept.name == spec.name:
                seen = True
        if not seen:
            out.append(spec.copy())
    return out^
