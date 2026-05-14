from std.utils import Variant
from std.memory import OwnedPointer
from std.sys.intrinsics import _type_is_eq

import marrow.dtypes as dt
from marrow.utils import variant_dispatch, variant_dispatch_raises


trait Value(Copyable, Movable, ImplicitlyDestructible):
    comptime OutType: dt.DataType

    def type(self) -> Self.OutType:
        ...

    def to_any(self) -> AnyValue:
        ...



trait Unary(Value):
    comptime ArgType: Value

    def __init__(out self, var arg: Self.ArgType):
        ...


trait Binary(Value):
    comptime LeftType: Value
    comptime RightType: Value

    def __init__(
        out self, var left: Self.LeftType, var right: Self.RightType
    ):
        ...


struct Column[T: dt.DataType](Value):
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

    def to_any(self) -> AnyValue:
        return AnyValue(Column(self.name, self.dtype.copy().to_any()))



struct Negate[T: Value](Unary):
    comptime ArgType = Self.T
    comptime OutType = Self.T.OutType

    var arg: Self.T

    def __init__(out self, var arg: Self.T):
        comptime assert conforms_to(Self.OutType, dt.NumericType), "Negate only supports numeric types"
        self.arg = arg^

    def __init__(out self, *, copy: Self):
        self.arg = copy.arg.copy()

    def type(self) -> Self.OutType:
        return self.arg.type()

    def to_any(self) -> AnyValue:
        return AnyValue(Negate(self.arg.to_any()))


struct Equal[L: Value, R: Value](Binary):
    comptime LeftType = Self.L
    comptime RightType = Self.R
    comptime OutType = dt.BoolType

    var left: Self.L
    var right: Self.R

    def __init__(out self, var left: Self.L, var right: Self.R):
        comptime assert _type_is_eq[Self.L.OutType, Self.R.OutType](), "Equal only supports comparing values of the same type"
        self.left = left^
        self.right = right^

    def __init__(out self, *, copy: Self):
        self.left = copy.left.copy()
        self.right = copy.right.copy()

    def type(self) -> Self.OutType:
        return dt.BoolType()

    def to_any(self) -> AnyValue:
        return AnyValue(Equal(self.left.to_any(), self.right.to_any()))


struct Add[L: Value, R: Value](Binary):
    comptime LeftType = Self.L
    comptime RightType = Self.R
    comptime OutType = Self.L.OutType

    var left: Self.L
    var right: Self.R

    def __init__(out self, var left: Self.L, var right: Self.R):
        comptime assert conforms_to(Self.L.OutType, dt.NumericType), "Add only supports numeric types"
        comptime assert _type_is_eq[Self.L.OutType, Self.R.OutType](), "Add only supports adding values of the same type"
        self.left = left^
        self.right = right^

    def __init__(out self, *, copy: Self):
        self.left = copy.left.copy()
        self.right = copy.right.copy()

    def type(self) -> Self.OutType:
        return self.left.type()

    def to_any(self) -> AnyValue:
        return AnyValue(Add(self.left.to_any(), self.right.to_any()))




struct AnyValue(Value):
    comptime OutType = dt.AnyDataType
    comptime VariantType = Variant[
        Column[dt.AnyDataType],
        Negate[AnyValue],
        Equal[AnyValue, AnyValue],
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

    def to_any(self) -> AnyValue:
        return self.copy()




def main() raises:
    var a = Column("a", dt.int64)
    var b = Column("b", dt.int64)
    var s = Column("s", dt.string)

    var c = Negate(a.copy())
    var d = Equal(a.copy(), b.copy())


    var f = d.to_any()
    var p = Add(a.copy(), b.copy())

    var t = f.type()
    var x = d.type()
    print("x:", x, "t:", t)

    print("Hello, world!")