"""Late-bound query parameters: cells, declarations and the registry.

A `ParamCell` is a shared, mutable box for a single scalar value that starts
unbound and is filled in later — after a plan has been built, once the caller
knows what to run it with. `ParamDecl` is the declaration a plan author writes
(name, dtype, optional default, optional help text) plus the `ArcPointer` to
the cell that expression nodes close over. `register_param` /
`drain_params` are a **module-level registry**, not a `parameters()` method on
every expression node: a `parameters()` sibling to `referenced_columns()` would
need 40 implementations and a second recursive plan traversal, against a size
gate where one shared dispatch adapter already cost +662,740 bytes (see
`DynScalar._dispatch`). Declaring a parameter appends to the registry as a
side effect of construction; the plan builder drains it once, after building
the tree that referenced the declarations.

`PathSpec` is the first consumer: a `ParquetScan` path that is either a literal
string or a cell to be resolved at execution time.

No expression node lives here yet — later tasks add nodes in `values.mojo` /
`dynamic.mojo` that hold a `ParamDecl`/`ArcPointer[ParamCell]` and import this
module. That is an expected cycle (this module does not import them back), not
one to design around.
"""

from std.utils import Variant
from std.memory import ArcPointer
from std.ffi import _Global

from ..dtypes import DynType
from ..scalars import DynScalar

# ---------------------------------------------------------------------------
# ParamCell
# ---------------------------------------------------------------------------


struct ParamCell(Copyable, Movable):
    """A shared, mutable box for one late-bound scalar value.

    Starts unbound (`get()` raises); `set()` binds it. Multiple expression
    nodes and the `ParamDecl` that declared it all share the same cell through
    an `ArcPointer`, so binding it once is visible everywhere it is read.
    """

    var _name: String
    var _value: Optional[DynScalar]

    def __init__(out self, var name: String = String()):
        self._name = name^
        self._value = None

    def get(self) raises -> DynScalar:
        """The bound value, or raises if this cell has not been bound yet.

        Does not name the parameter — the cell alone does not know it is
        being read through a `ParamDecl`; `ParamDecl` is where a caller-facing,
        named error belongs.
        """
        if not self._value:
            raise Error("parameter is not bound")
        return self._value.value().copy()

    def set(mut self, var v: DynScalar):
        self._value = v^

    def is_bound(self) -> Bool:
        return Bool(self._value)

    def name_hint(self) -> String:
        """The name this cell was constructed with, for use in rendering
        (`NumericParam.render`) — not authoritative once a cell outlives its
        declaring `ParamDecl`."""
        return self._name


# ---------------------------------------------------------------------------
# ParamDecl
# ---------------------------------------------------------------------------


struct ParamDecl(Copyable, Movable):
    """A parameter declaration: name, dtype, optional help text and default,
    plus the cell expression nodes bind to.

    Constructing one registers no side effect by itself — `register_param`
    does that, explicitly, at the call site that declares a plan parameter.
    """

    var name: String
    var dtype: DynType
    var help: String
    var default: Optional[DynScalar]
    var cell: ArcPointer[ParamCell]

    def __init__(
        out self,
        *,
        var name: String,
        dtype: DynType,
        var help: String = String(),
        var default: Optional[DynScalar] = None,
        cell: Optional[ArcPointer[ParamCell]] = None,
    ):
        self.dtype = dtype.copy()
        self.help = help^
        self.default = default^
        if cell:
            self.cell = cell.value().copy()
        else:
            self.cell = ArcPointer(ParamCell(name.copy()))
        self.name = name^

    def is_required(self) -> Bool:
        return not self.default


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _init_registry() -> List[ParamDecl]:
    return List[ParamDecl]()


comptime _REGISTRY = _Global["MARROW_PARAM_REGISTRY", _init_registry]
"""The process-wide, in-flight parameter registry.

A plan-building call that declares a parameter (a later task's `param()`
factory) appends a `ParamDecl` here; the plan builder drains the whole list
once it has finished building the tree the declarations belong to. Draining
resets the registry to empty so the *next* plan starts clean — and to exactly
the set a later `lookup_param(name)` name-keyed table gets reset to, since that
table is rebuilt from the same drained `List[ParamDecl]`.

`get_or_create_ptr()` only raises when `_Global`'s `on_error_msg` is set, which
it is not here, so the `except` arms below are unreachable in practice; they
exist so `register_param`/`drain_params` keep the non-raising signatures a
registry-as-side-effect API needs.
"""


def register_param(var decl: ParamDecl):
    """Append `decl` to the module-level registry."""
    try:
        _REGISTRY.get_or_create_ptr()[].append(decl^)
    except:
        pass


def drain_params() -> List[ParamDecl]:
    """Return the registry's contents and empty it."""
    try:
        var ptr = _REGISTRY.get_or_create_ptr()
        var out = ptr[].copy()
        ptr[] = List[ParamDecl]()
        return out^
    except:
        return List[ParamDecl]()


# ---------------------------------------------------------------------------
# PathSpec
# ---------------------------------------------------------------------------


struct PathSpec(Copyable, Movable):
    """A `ParquetScan` path: either a literal string or a cell resolved at
    execution time."""

    comptime VariantType = Variant[String, ArcPointer[ParamCell]]
    var _v: Self.VariantType

    @implicit
    def __init__(out self, var path: String):
        self._v = Self.VariantType(path^)

    def __init__(out self, cell: ArcPointer[ParamCell]):
        self._v = Self.VariantType(cell)

    def resolve(self) raises -> String:
        if self._v.isa[String]():
            return self._v[String]
        else:
            return self._v[ArcPointer[ParamCell]][].get().as_string().to_string()
