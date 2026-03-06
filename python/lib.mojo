from std.python import PythonObject, Python
from std.python.bindings import PythonModuleBuilder
from std.os import abort

from marrow.module.dtypes_api import add_to_module as add_dtypes
from marrow.module.arrays.primitive_api import add_to_module as add_primitive


@export
fn PyInit_marrow() -> PythonObject:
    try:
        var m = PythonModuleBuilder("marrow")
        add_dtypes(m)
        add_primitive(m)
        return m.finalize()
    except e:
        abort(String("error creating Python Mojo module:", e))
