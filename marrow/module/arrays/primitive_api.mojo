"""Python interface for primitive array."""

from std.os import abort
from std.python import Python
from std.python.bindings import PythonModuleBuilder, PythonObject

from marrow.dtypes import DataType, int64
from marrow.arrays.base import ArrayData
from marrow.arrays import primitive


@fieldwise_init
struct PrimitiveArray(Movable, Writable):
    """Type erased PrimitiveArray so that we can return to python."""

    var data: ArrayData
    var offset: Int
    var capacity: Int

    fn write_to(self, mut writer: Some[Writer]):
        writer.write("PrimitiveArray")
    
    fn write_repr_to(self, mut writer: Some[Writer]):
        writer.write("PrimitiveArray")

    @staticmethod
    fn __len__(py_self: PythonObject) raises -> PythonObject:
        """Return the length of the underlying ArrayData."""
        var self_ptr = py_self.downcast_value_ptr[Self]()
        return self_ptr[].data.length

    @staticmethod
    fn __getitem__(
        py_self: PythonObject, index: PythonObject
    ) raises -> PythonObject:
        """Access the element at the given index."""
        var self_ptr = py_self.downcast_value_ptr[Self]()
        return PythonObject(
            primitive.Int64Array(self_ptr[].data.copy()).unsafe_get(
                Int(py=index)
            )
        )


fn array(content: PythonObject) raises -> PythonObject:
    """Create a primitive array, only In64 implemented so far.

    Args:
        content: An iterable of Ints.

    Returns:
        A PrimitiveArray wrapped in a PythonObject.

    """
    var actual = primitive.Int64Array()

    for v in content:
        actual.append(rebind[Scalar[int64.native]](Int64(py=v)))

    var result = PrimitiveArray(
        data=actual.data.copy(),
        offset=actual.offset,
        capacity=actual.capacity,
    )
    return PythonObject(alloc=result^)


def add_to_module(mut builder: PythonModuleBuilder) raises:
    """Add primitive array support to the python API."""

    _ = builder.add_type[PrimitiveArray]("PrimitiveArray")
        .def_method[PrimitiveArray.__len__]("__len__")
        .def_method[PrimitiveArray.__getitem__]("__getitem__")
    
    builder.def_function[array](
        "array",
        docstring="Build a primitive array with the given data and datatype",
    )
