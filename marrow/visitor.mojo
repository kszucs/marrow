from .arrays import *
from .dtypes import *


trait DataTypeVisitor:
    """Dispatch operations based on a runtime DataType value.

    Implement this trait and call `visitor.visit(dtype)` to receive a
    dispatch to the appropriate typed overload.

    `visit[T: DataType]` is invoked for primitive types (bool, int*, uint*,
    float*). `visit_string` is invoked for string. Nested type overloads
    (`visit_list`, `visit_fixed_size_list`, `visit_struct`) raise by default.
    `visit(DataType)` dispatches to the typed overloads by runtime dtype.
    """

    def visit[T: DataType](mut self) raises:
        pass

    def visit_string(mut self) raises:
        pass

    def visit_binary(mut self) raises:
        raise Error("visit_binary: not implemented")

    def visit_list(mut self, child: DataType) raises:
        raise Error("visit_list: not implemented")

    def visit_fixed_size_list(mut self, child: DataType, size: Int) raises:
        raise Error("visit_fixed_size_list: not implemented")

    def visit_struct(mut self, fields: List[Field]) raises:
        raise Error("visit_struct: not implemented")

    def visit(mut self, dtype: DataType) raises:
        """Dispatch to the typed overload matching the runtime dtype."""
        comptime for dt in primitive_dtypes:
            if dtype == dt:
                self.visit[dt]()
                return

        if dtype.is_string():
            self.visit_string()
        elif dtype.is_binary():
            self.visit_binary()
        elif dtype.is_list():
            self.visit_list(dtype.fields[0].dtype)
        elif dtype.is_fixed_size_list():
            self.visit_fixed_size_list(dtype.fields[0].dtype, dtype.size)
        elif dtype.is_struct():
            self.visit_struct(dtype.fields)
        else:
            raise Error("visit: unsupported dtype: ", dtype)


trait ArrayVisitor:
    """Trait for type-dispatched array operations.

    Implement this trait and call `visitor.visit(array)` to receive a
    concretely-typed array matching the runtime dtype of the AnyArray.

    All typed `visit` overloads have default no-op bodies, so implementors
    only need to override the array kinds they care about. `visit(AnyArray)`
    dispatches to the typed overloads by default. `visit(ChunkedArray)`
    dispatches to each chunk by default.

    Typed arrays passed to visitor methods share the underlying buffer memory
    with the original AnyArray (ArcPointer semantics). Visitor methods are
    `raises` to allow implementations that perform I/O or recursive dispatch
    into nested arrays.
    """

    def visit[T: DataType](mut self, array: PrimitiveArray[T]) raises:
        pass

    def visit(mut self, array: StringArray) raises:
        pass

    def visit(mut self, array: ListArray) raises:
        pass

    def visit(mut self, array: FixedSizeListArray) raises:
        pass

    def visit(mut self, array: StructArray) raises:
        pass

    def visit(mut self, array: ChunkedArray) raises:
        for chunk in array.chunks:
            self.visit(chunk)

    def visit(mut self, array: AnyArray) raises:
        """Dispatch to the typed overload matching the runtime dtype."""

        comptime for dtype in primitive_dtypes:
            if array.dtype == dtype:
                self.visit[dtype](PrimitiveArray[dtype](data=array))
                return

        if array.dtype.is_string():
            self.visit(StringArray(data=array))
        elif array.dtype.is_list():
            self.visit(ListArray(data=array))
        elif array.dtype.is_fixed_size_list():
            self.visit(FixedSizeListArray(data=array))
        elif array.dtype.is_struct():
            self.visit(StructArray(data=array))
        else:
            raise Error("visit: unsupported dtype ", array.dtype)
