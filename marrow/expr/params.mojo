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

The fused lane's param nodes (`NumericParam`/`StringParam`/`TemporalParam` in
`values.mojo`) hold their `ArcPointer[ParamCell]` directly, since they are
built with a known dtype and can close over the cell at construction time. The
runtime lane's `DynValue.param` leaf cannot: `DynPayload` is size-critical
(see `dynamic.mojo`) and carries only the parameter's *name*, so its evaluator
resolves the cell by name at execute time instead, via `lookup_param`.

That name-keyed lookup is a **second module-level table** (`_LOOKUP`),
separate from `_REGISTRY`, and the two are reset on an intentionally
asymmetric schedule: **`drain_params()` empties `_REGISTRY` but
*(re-)populates* `_LOOKUP`** with exactly the declarations it just drained.
A plan's drain therefore opens that plan's own name-resolution scope for the
runtime lane — `lookup_param` keeps working *after* the drain, which is the
point, since `_param` runs at execute time, well after the tree was built —
while a later, unrelated plan's drain replaces the scope rather than adding to
it, so one process building two plans back-to-back never resolves a name from
the wrong plan. Registration is last-wins per name for the same reason: a
name declared twice before a drain (deliberately or via a rebuilt plan)
leaves `_LOOKUP` pointing at the most recent cell, not an arbitrary one.

No expression node lives here — nodes in `values.mojo` / `dynamic.mojo` hold a
`ParamDecl`/`ArcPointer[ParamCell]` and import this module. `PathSpec` closes
the loop: its `StringParam[T]` constructor overload imports `StringParam` back
from `values.mojo` to share its cell, so the two modules import each other.
That is an expected cycle in this package (see CLAUDE.md), not one to design
around.
"""

from std.utils import Variant
from std.memory import ArcPointer
from std.ffi import _Global

from ..dtypes import DynType, StringLikeType
from ..scalars import DynScalar
from .values import StringParam

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


def _init_lookup() -> Dict[String, ArcPointer[ParamCell]]:
    return Dict[String, ArcPointer[ParamCell]]()


comptime _LOOKUP = _Global["MARROW_PARAM_LOOKUP", _init_lookup]
"""The runtime lane's name-keyed parameter table.

`register_param` upserts into this as well as into `_REGISTRY`, and
`drain_params` rebuilds it from exactly the declarations it drains — see the
module docstring for the asymmetry (registry empties, lookup repopulates).
`lookup_param` is the only reader."""


def register_param(var decl: ParamDecl):
    """Append `decl` to the module-level registry, and upsert its cell into
    `_LOOKUP` under its name — last-wins, so a name declared twice before the
    next drain points `lookup_param` at the most recently declared cell."""
    try:
        _LOOKUP.get_or_create_ptr()[][decl.name.copy()] = decl.cell.copy()
    except:
        pass
    try:
        _REGISTRY.get_or_create_ptr()[].append(decl^)
    except:
        pass


def drain_params() -> List[ParamDecl]:
    """Return the registry's contents and empty it.

    Also resets `_LOOKUP` to exactly the set just drained — see the module
    docstring for why that reset is a *repopulation*, not a clear."""
    try:
        var ptr = _REGISTRY.get_or_create_ptr()
        var out = ptr[].copy()
        ptr[] = List[ParamDecl]()
        try:
            var lookup = _LOOKUP.get_or_create_ptr()
            lookup[].clear()
            for ref decl in out:
                lookup[][decl.name.copy()] = decl.cell.copy()
        except:
            pass
        return out^
    except:
        return List[ParamDecl]()


def lookup_param(name: String) raises -> ParamCell:
    """Resolve a runtime-lane parameter by name, against `_LOOKUP` — the
    table `drain_params()` last populated, not `_REGISTRY`, which is empty
    between drains by design.

    `DynValue.param`'s payload carries only the name (`DynPayload` gained no
    new arm for parameters), so `DynValue._param` calls this once per batch,
    at evaluate time, to find the cell a plan author bound after building the
    tree."""
    var found = _LOOKUP.get_or_create_ptr()[].get(name)
    if found:
        return found.value()[].copy()
    else:
        raise Error("unknown parameter: " + name)


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

    @implicit
    def __init__[T: StringLikeType](out self, param: StringParam[T]):
        """Build from a fused-lane string parameter — `ParquetScan(path=src,
        ...)` for `src = param("src", string)`. Shares `param`'s cell rather
        than copying its bound value, so binding the parameter after building
        the plan (the whole point of a late-bound path) is visible here too.
        """
        self._v = Self.VariantType(param.cell())

    def resolve(self) raises -> String:
        if self._v.isa[String]():
            return self._v[String]
        else:
            return self._v[ArcPointer[ParamCell]][].get().as_string().to_string()

    def describe(self) -> String:
        """Non-raising rendering for plan display (`ParquetScan.write_to`):
        the literal path, or `param(name)` for a still-unbound cell — mirrors
        `StringParam.render()`, since `resolve()` cannot be called from a
        non-raising `write_to`."""
        if self._v.isa[String]():
            return self._v[String]
        else:
            return String(
                "param(", self._v[ArcPointer[ParamCell]][].name_hint(), ")"
            )
