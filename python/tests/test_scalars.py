"""Test Python scalar types.

Covers:
  - Scalar from aggregate functions (sum_, product, min_, max_)
  - Scalar from array __getitem__ (int, float, bool, string)
  - as_py(), is_valid(), type(), __str__, __repr__
  - Rich comparison (__eq__, __ne__, __lt__, __le__, __gt__, __ge__)
"""

import marrow as ma


# -- scalar from aggregates --------------------------------------------------


def test_scalar_from_sum():
    s = ma.sum_(ma.array([1, 2, 3, 4]))
    assert type(s).__name__ == "Scalar"
    assert s == 10
    assert s == 10.0
    assert s.as_py() == 10
    assert s.is_valid()


def test_scalar_from_product():
    s = ma.product(ma.array([2, 3, 4]))
    assert s == 24
    assert s.as_py() == 24


def test_scalar_from_min():
    s = ma.min_(ma.array([5, 1, 3]))
    assert s == 1
    assert s.as_py() == 1


def test_scalar_from_max():
    s = ma.max_(ma.array([5, 1, 3]))
    assert s == 5
    assert s.as_py() == 5


# -- scalar from array indexing ----------------------------------------------


def test_scalar_from_int64_getitem():
    arr = ma.array([10, 20, 30])
    s = arr[0]
    assert type(s).__name__ == "Scalar"
    assert s == 10
    assert s.as_py() == 10
    assert s.is_valid()


def test_scalar_from_float64_getitem():
    arr = ma.array([1.5, 2.5, 3.5])
    assert arr[0] == 1.5
    assert arr[2] == 3.5
    assert arr[1].as_py() == 2.5


def test_scalar_from_bool_getitem():
    arr = ma.array([True, False, True])
    assert arr[0] == True
    assert arr[1] == False


def test_scalar_from_string_getitem():
    arr = ma.array(["hello", "world"])
    assert arr[0] == "hello"
    assert arr[1] == "world"
    assert arr[0].as_py() == "hello"


# -- scalar methods ----------------------------------------------------------


def test_scalar_str():
    s = ma.sum_(ma.array([1, 2, 3]))
    assert str(s) == "6"


def test_scalar_repr():
    s = ma.sum_(ma.array([1, 2, 3]))
    assert "6" in repr(s)


def test_scalar_is_valid():
    arr = ma.array([1, 2, 3])
    assert arr[0].is_valid()
    assert not arr[0].is_null()


def test_scalar_type():
    s = ma.sum_(ma.array([1, 2, 3], type=ma.int64()))
    assert str(s.type()) == "int64"


# -- comparison operators ----------------------------------------------------


def test_scalar_eq_ne():
    s = ma.sum_(ma.array([1, 2, 3, 4]))
    assert s == 10
    assert s != 5
    assert not (s == 5)
    assert not (s != 10)


def test_scalar_lt_le_gt_ge():
    s = ma.sum_(ma.array([1, 2, 3, 4]))
    assert s > 5
    assert s >= 10
    assert s < 20
    assert s <= 10
    assert not (s < 10)
    assert not (s > 10)


def test_scalar_cross_type_comparison():
    """int scalar == float value and vice versa."""
    s = ma.sum_(ma.array([1, 2, 3, 4]))
    assert s == 10.0
    assert s != 10.1
