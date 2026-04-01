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

    def visit_list(mut self, child: ArrowType) raises:
        raise Error("visit_list: not implemented")

    def visit_fixed_size_list(mut self, child: ArrowType, size: Int) raises:
        raise Error("visit_fixed_size_list: not implemented")

    def visit_struct(mut self, fields: List[Field]) raises:
        raise Error("visit_struct: not implemented")

    def visit(mut self, dtype: ArrowType) raises:
        """Dispatch to the typed overload matching the runtime dtype."""
        if dtype == bool_:
            self.visit[BoolType]()
            return
        elif dtype == int8:
            self.visit[Int8Type]()
            return
        elif dtype == int16:
            self.visit[Int16Type]()
            return
        elif dtype == int32:
            self.visit[Int32Type]()
            return
        elif dtype == int64:
            self.visit[Int64Type]()
            return
        elif dtype == uint8:
            self.visit[UInt8Type]()
            return
        elif dtype == uint16:
            self.visit[UInt16Type]()
            return
        elif dtype == uint32:
            self.visit[UInt32Type]()
            return
        elif dtype == uint64:
            self.visit[UInt64Type]()
            return
        elif dtype == float16:
            self.visit[Float16Type]()
            return
        elif dtype == float32:
            self.visit[Float32Type]()
            return
        elif dtype == float64:
            self.visit[Float64Type]()
            return

        if dtype.is_string():
            self.visit_string()
        elif dtype.is_binary():
            self.visit_binary()
        elif dtype.is_list():
            self.visit_list(dtype.as_list_type().item[])
        elif dtype.is_fixed_size_list():
            var fsl = dtype.as_fixed_size_list_type()
            self.visit_fixed_size_list(fsl.item.dtype[], fsl.size)
        elif dtype.is_struct():
            self.visit_struct(dtype.as_struct_type().fields)
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

    def visit[T: PrimitiveType](mut self, array: PrimitiveArray[T]) raises:
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
        var dt = array.dtype()

        if dt == bool_:
            self.visit[BoolType](array.as_primitive[BoolType]())
            return
        elif dt == int8:
            self.visit[Int8Type](array.as_primitive[Int8Type]())
            return
        elif dt == int16:
            self.visit[Int16Type](array.as_primitive[Int16Type]())
            return
        elif dt == int32:
            self.visit[Int32Type](array.as_primitive[Int32Type]())
            return
        elif dt == int64:
            self.visit[Int64Type](array.as_primitive[Int64Type]())
            return
        elif dt == uint8:
            self.visit[UInt8Type](array.as_primitive[UInt8Type]())
            return
        elif dt == uint16:
            self.visit[UInt16Type](array.as_primitive[UInt16Type]())
            return
        elif dt == uint32:
            self.visit[UInt32Type](array.as_primitive[UInt32Type]())
            return
        elif dt == uint64:
            self.visit[UInt64Type](array.as_primitive[UInt64Type]())
            return
        elif dt == float16:
            self.visit[Float16Type](array.as_primitive[Float16Type]())
            return
        elif dt == float32:
            self.visit[Float32Type](array.as_primitive[Float32Type]())
            return
        elif dt == float64:
            self.visit[Float64Type](array.as_primitive[Float64Type]())
            return

        if dt.is_string():
            self.visit(array.as_string())
        elif dt.is_list():
            self.visit(array.as_list())
        elif dt.is_fixed_size_list():
            self.visit(array.as_fixed_size_list())
        elif dt.is_struct():
            self.visit(array.as_struct())
        else:
            raise Error("visit: unsupported dtype ", dt)
