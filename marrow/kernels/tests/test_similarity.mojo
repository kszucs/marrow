"""Tests for batch cosine similarity kernel."""

from std.testing import assert_equal, assert_true, TestSuite
from marrow.arrays import AnyArray, PrimitiveArray, FixedSizeListArray
from marrow.builders import AnyBuilder, PrimitiveBuilder, FixedSizeListBuilder
from marrow.dtypes import float32, fixed_size_list_
from marrow.kernels.similarity import cosine_similarity


def _make_vectors(*values: Float64, dim: Int) raises -> FixedSizeListArray:
    """Helper: build FixedSizeListArray from flat values."""
    var b = PrimitiveBuilder[float32](len(values))
    for v in values:
        b.append(Scalar[float32.native](v))
    var n_lists = len(values) // dim
    var builder = FixedSizeListBuilder(AnyBuilder(b^), list_size=dim)
    for _ in range(n_lists):
        builder.append_valid()
    return builder.finish_typed()


def _make_query(*values: Float64) raises -> PrimitiveArray[float32]:
    """Helper: build query PrimitiveArray."""
    var b = PrimitiveBuilder[float32](len(values))
    for v in values:
        b.append(Scalar[float32.native](v))
    return b.finish_typed()


# TODO: use abs and place this to a common test utils module
def _approx_equal(a: Float64, b: Float64, tol: Float64 = 1e-5) -> Bool:
    var diff = a - b
    if diff < 0:
        diff = -diff
    return diff < tol


def test_cosine_similarity_identical() raises:
    """Identical vectors → score ≈ 1.0."""
    var vectors = _make_vectors(1.0, 2.0, 3.0, dim=3)
    var query = _make_query(1.0, 2.0, 3.0)
    var scores = cosine_similarity[float32](vectors, query)
    assert_equal(len(scores), 1)
    assert_true(_approx_equal(Float64(scores.unsafe_get(0)), 1.0))


def test_cosine_similarity_opposite() raises:
    """Opposite vectors → score ≈ -1.0."""
    var vectors = _make_vectors(-1.0, -2.0, -3.0, dim=3)
    var query = _make_query(1.0, 2.0, 3.0)
    var scores = cosine_similarity[float32](vectors, query)
    assert_equal(len(scores), 1)
    assert_true(_approx_equal(Float64(scores.unsafe_get(0)), -1.0))


def test_cosine_similarity_orthogonal() raises:
    """Orthogonal vectors → score ≈ 0.0."""
    var vectors = _make_vectors(1.0, 0.0, dim=2)
    var query = _make_query(0.0, 1.0)
    var scores = cosine_similarity[float32](vectors, query)
    assert_equal(len(scores), 1)
    assert_true(_approx_equal(Float64(scores.unsafe_get(0)), 0.0))


def test_cosine_similarity_batch() raises:
    """Multiple vectors: identical, opposite, orthogonal."""
    # 3 vectors of dim 2: [1,0], [-1,0], [0,1]
    var vectors = _make_vectors(1.0, 0.0, -1.0, 0.0, 0.0, 1.0, dim=2)
    var query = _make_query(1.0, 0.0)
    var scores = cosine_similarity[float32](vectors, query)
    assert_equal(len(scores), 3)
    assert_true(_approx_equal(Float64(scores.unsafe_get(0)), 1.0))
    assert_true(_approx_equal(Float64(scores.unsafe_get(1)), -1.0))
    assert_true(_approx_equal(Float64(scores.unsafe_get(2)), 0.0))


def test_cosine_similarity_zero_vector() raises:
    """Zero vector → score = 0.0 (avoid division by zero)."""
    var vectors = _make_vectors(0.0, 0.0, 0.0, dim=3)
    var query = _make_query(1.0, 2.0, 3.0)
    var scores = cosine_similarity[float32](vectors, query)
    assert_equal(len(scores), 1)
    assert_true(_approx_equal(Float64(scores.unsafe_get(0)), 0.0))


def test_cosine_similarity_dimension_mismatch() raises:
    """Query dim != vector dim raises."""
    var vectors = _make_vectors(1.0, 2.0, 3.0, dim=3)
    var query = _make_query(1.0, 2.0)
    try:
        _ = cosine_similarity[float32](vectors, query)
        assert_true(False, "should have raised")
    except:
        pass


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
