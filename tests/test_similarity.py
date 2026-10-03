import pytest

from knowledge_agent2_0.store.similarity import cosine_similarity


def test_identical_vectors():
    vector = [1.0, 0.0, 0.0]

    assert cosine_similarity(vector, vector) == pytest.approx(1.0)


def test_orthogonal_vectors():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_opposite_vectors():
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_scaled_vectors():
    assert cosine_similarity([1.0, 1.0], [3.0, 3.0]) == pytest.approx(1.0)


def test_zero_vector_returns_zero():
    assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


def test_mismatched_dimensions_raise():
    with pytest.raises(ValueError, match="Embedding dimensions do not match"):
        cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])
