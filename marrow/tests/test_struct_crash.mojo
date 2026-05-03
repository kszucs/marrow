from std.testing import assert_equal, assert_true
from marrow.testing import TestSuite
from marrow.builders import StructBuilder
from marrow.arrays import AnyArray, StructArray
from marrow.scalars import AnyScalar
from marrow.dtypes import int32, int64, field, Int32Type, Int64Type

def _make_struct_arr() raises -> StructArray:
    var sb = StructBuilder([field("x", int32), field("y", int64)], capacity=1)
    sb.field_builder(0).as_int32().append(1)
    sb.field_builder(1).as_int64().append(10)
    sb.append_valid()
    return sb.finish()

# Access children directly — bypasses StructArray.__getitem__
def test_child0_direct() raises:
    var arr = _make_struct_arr()
    var child0: AnyArray = arr.children[0].copy()
    var s = child0[0]
    assert_equal(s.as_int32().value(), 1)

def test_child1_direct() raises:
    var arr = _make_struct_arr()
    var child1: AnyArray = arr.children[1].copy()
    var s = child1[0]
    assert_equal(s.as_int64().value(), 10)

# Build List[AnyScalar] manually the same way StructArray.__getitem__ does
def test_build_fields_list() raises:
    var arr = _make_struct_arr()
    var fields = List[AnyScalar]()
    for ref child in arr.children:
        fields.append(child[0])
    assert_equal(len(fields), 2)

def test_fields_list_index0() raises:
    var arr = _make_struct_arr()
    var fields = List[AnyScalar]()
    for ref child in arr.children:
        fields.append(child[0])
    assert_equal(fields[0].as_int32().value(), 1)

def test_fields_list_index1() raises:
    var arr = _make_struct_arr()
    var fields = List[AnyScalar]()
    for ref child in arr.children:
        fields.append(child[0])
    assert_equal(fields[1].as_int64().value(), 10)

def main() raises:
    TestSuite.run[__functions_in_module()]()
