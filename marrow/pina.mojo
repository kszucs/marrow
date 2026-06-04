from std.utils import Variant
from std.memory import OwnedPointer
from std.sys.intrinsics import _type_is_eq

import marrow.dtypes as dt
from marrow.utils import variant_dispatch, variant_dispatch_raises


trait Value(Copyable, ImplicitlyDestructible, Movable, Writable):
    comptime OutType: dt.DataType

    def type(self) -> Self.OutType:
        ...

    def to_any(self) raises -> AnyValue:
        ...


trait NumericValue(Value):
    """Marker trait for expression nodes whose OutType is numeric.
    Implementors must satisfy: conforms_to(Self.OutType, dt.NumericType).
    """

    def negate(self) raises -> Negate[Self]:
        return Negate(self.copy())


trait Unary(Value):
    comptime ArgType: Value

    def __init__(out self, var arg: Self.ArgType) raises:
        ...


trait Binary(Value):
    comptime LeftType: Value
    comptime RightType: Value

    def __init__(
        out self, var left: Self.LeftType, var right: Self.RightType
    ) raises:
        ...


struct Column[T: dt.DataType](
    NumericValue where conforms_to(T, dt.NumericType),
    Value,
):
    comptime OutType = Self.T

    var name: String
    var dtype: Self.T

    def __init__(out self, name: String, var dtype: Self.T):
        self.name = name
        self.dtype = dtype^

    def __init__(out self, *, copy: Self):
        self.name = copy.name
        self.dtype = copy.dtype.copy()

    def type(self) -> Self.OutType:
        return self.dtype.copy()

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self.name)

    def to_any(self) -> AnyValue:
        return AnyValue(Column(self.name, self.dtype.copy().to_any()))


struct Negate[T: NumericValue](NumericValue, Unary):
    comptime ArgType = Self.T
    comptime OutType = Self.T.OutType

    var arg: Self.T

    def __init__(out self, var arg: Self.T) raises:
        if not arg.type().to_any().is_numeric():
            raise Error(
                "Negate only supports numeric types, got: " + String(arg.type())
            )
        self.arg = arg^

    def __init__(out self, *, copy: Self):
        self.arg = copy.arg.copy()

    def type(self) -> Self.OutType:
        return self.arg.type()

    def write_to[W: Writer](self, mut writer: W):
        writer.write("-(", self.arg, ")")

    def to_any(self) raises -> AnyValue:
        return AnyValue(Negate(self.arg.to_any()))


struct Equal[L: Value, R: Value](Binary):
    comptime LeftType = Self.L
    comptime RightType = Self.R
    comptime OutType = dt.BoolType

    var left: Self.L
    var right: Self.R

    def __init__(out self, var left: Self.L, var right: Self.R) raises:
        comptime assert _type_is_eq[
            Self.L.OutType, Self.R.OutType
        ](), "Equal only supports comparing values of the same type"
        self.left = left^
        self.right = right^

    def __init__(out self, *, copy: Self):
        self.left = copy.left.copy()
        self.right = copy.right.copy()

    def type(self) -> Self.OutType:
        return dt.BoolType()

    def write_to[W: Writer](self, mut writer: W):
        writer.write("(", self.left, " == ", self.right, ")")

    def to_any(self) raises -> AnyValue:
        return AnyValue(Equal(self.left.to_any(), self.right.to_any()))


struct Add[L: Value, R: Value](
    Binary,
    NumericValue where conforms_to(L, NumericValue) and conforms_to(
        R, NumericValue
    ),
):
    comptime LeftType = Self.L
    comptime RightType = Self.R
    comptime OutType = Self.L.OutType

    var left: Self.L
    var right: Self.R

    def __init__(out self, var left: Self.L, var right: Self.R) raises:
        if not left.type().to_any().is_numeric():
            raise Error(
                "Add only supports numeric types, got: " + String(left.type())
            )
        if left.type().to_any() != right.type().to_any():
            raise Error(
                "Add requires matching types, got: "
                + String(left.type())
                + " and "
                + String(right.type())
            )
        self.left = left^
        self.right = right^

    def __init__(out self, *, copy: Self):
        self.left = copy.left.copy()
        self.right = copy.right.copy()

    def type(self) -> Self.OutType:
        return self.left.type()

    def write_to[W: Writer](self, mut writer: W):
        writer.write("(", self.left, " + ", self.right, ")")

    def to_any(self) raises -> AnyValue:
        return AnyValue(Add(self.left.to_any(), self.right.to_any()))


struct AnyValue(NumericValue, Value):
    comptime OutType = dt.AnyDataType
    comptime VariantType = Variant[
        Column[dt.AnyDataType],
        Negate[AnyValue],
        Equal[AnyValue, AnyValue],
        Add[AnyValue, AnyValue],
    ]

    var value: OwnedPointer[Self.VariantType]

    def __init__(out self, var value: Self.VariantType):
        self.value = OwnedPointer[Self.VariantType](value^)

    def __init__(out self, *, copy: Self):
        var inner = Self.VariantType(copy=copy.value[])
        self.value = OwnedPointer[Self.VariantType](inner^)

    def type(self) -> Self.OutType:
        @parameter
        def f[T: Value](a: T) -> dt.AnyDataType:
            return a.type().to_any()

        return variant_dispatch[Value, func=f](self.value[])

    def write_to[W: Writer](self, mut writer: W):
        @parameter
        def f[T: Value](a: T):
            a.write_to(writer)

        variant_dispatch[Value, func=f](self.value[])

    def negate(self) raises -> Negate[Self]:
        if not self.type().is_numeric():
            raise Error(
                "negate only supports numeric types, got: "
                + String(self.type())
            )
        return Negate(self.copy())

    def to_any(self) -> AnyValue:
        return self.copy()


def negate[T: NumericValue](var arg: Negate[T]) -> T:
    return arg.arg.copy()


def negate[
    T: NumericValue
](var arg: T) raises -> Negate[T] where not conforms_to(T, Unary):
    return Negate(arg^)


def main() raises:
    var a = Column("a", dt.int64)
    var b = Column("b", dt.int64)

    var neg_a = a.negate()
    var double_neg = a.negate().negate()
    var eq = Equal(a.copy(), b.copy())
    var add = Add(a.copy(), b.copy())
    var nested = Equal(Add(a.copy(), b.copy()), Add(a.copy(), b.copy()))

    print(neg_a)
    print(double_neg)
    print(eq)
    print(add)
    print(nested)
    print(nested.to_any())
