"""Python-exposed Device type wrapping ExecutionContext.

Exposes the GPU execution context to Python callers. Construct with ``Device()``
to obtain the default Metal/CUDA device, then pass to ``Array.to_device()``,
``Array.to_cpu()``, and GPU-capable compute functions.
"""

from std.gpu.host import DeviceContext
from std.python import PythonObject, ConvertibleFromPython, ConvertibleToPython
from std.python.bindings import PythonModuleBuilder
from marrow.kernels.execution import ExecutionContext


struct Device(
    ConvertibleFromPython, ConvertibleToPython, Copyable, Movable, Writable
):
    """GPU device context handle.

    Wraps ``ExecutionContext`` for use from Python.  Construct with no arguments
    to obtain the default device::

        device = ma.Device()

    Pass to ``Array.to_device(device)`` to upload data, to ``Array.to_cpu(device)``
    to download results, and as the last argument to GPU-capable compute functions.
    """

    var ctx: ExecutionContext

    def __init__(out self) raises:
        self.ctx = ExecutionContext.gpu(DeviceContext())

    def __init__(out self, *, copy: Self):
        self.ctx = copy.ctx.copy()

    def __init__(out self, *, py: PythonObject) raises:
        self = py.downcast_value_ptr[Device]()[].copy()

    def write_to[W: Writer](self, mut writer: W):
        writer.write("Device()")

    def write_repr_to[W: Writer](self, mut writer: W):
        self.write_to(writer)

    def to_python_object(var self) raises -> PythonObject:
        return PythonObject(alloc=self^)


def add_to_module(mut mb: PythonModuleBuilder) raises -> None:
    """Register the Device type in the marrow Python module."""
    _ = mb.add_type[Device]("Device")
