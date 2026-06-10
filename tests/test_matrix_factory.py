"""Tests for synthetic matrix generation."""

import numpy as np
import pytest

from quant.matrix_factory import (
    MatrixKind,
    gaussian_matrix,
    heavy_tailed_matrix,
    make_matrix,
    outlier_matrix,
)


def test_gaussian_matrix_is_reproducible() -> None:
    first = gaussian_matrix((8, 4), seed=7)
    second = gaussian_matrix((8, 4), seed=7)

    np.testing.assert_array_equal(first, second)


def test_gaussian_matrix_shape_and_dtype() -> None:
    matrix = gaussian_matrix((3, 5), seed=1, dtype=np.float64)

    assert matrix.shape == (3, 5)
    assert matrix.dtype == np.float64


def test_heavy_tailed_matrix_is_reproducible() -> None:
    first = heavy_tailed_matrix((6, 6), df=1.5, seed=11)
    second = heavy_tailed_matrix((6, 6), df=1.5, seed=11)

    np.testing.assert_array_equal(first, second)


def test_outlier_matrix_injects_expected_number_of_large_values() -> None:
    matrix = outlier_matrix(
        (10, 10),
        base_mean=0.0,
        base_std=1.0,
        outlier_fraction=0.1,
        outlier_scale=20.0,
        seed=13,
    )

    large_values = np.abs(matrix) > 10.0
    assert large_values.sum() == 10


def test_make_matrix_dispatches_from_string_kind() -> None:
    matrix = make_matrix("gaussian", (2, 3), seed=3)

    assert matrix.shape == (2, 3)


def test_make_matrix_dispatches_from_enum_kind() -> None:
    matrix = make_matrix(MatrixKind.HEAVY_TAILED, (2, 3), seed=3)

    assert matrix.shape == (2, 3)


@pytest.mark.parametrize(
    ("factory", "kwargs", "message"),
    [
        (gaussian_matrix, {"shape": (0, 3)}, "shape dimensions"),
        (gaussian_matrix, {"shape": (3, 3), "std": 0.0}, "std"),
        (heavy_tailed_matrix, {"shape": (3, 3), "df": 0.0}, "df"),
        (heavy_tailed_matrix, {"shape": (3, 3), "scale": 0.0}, "scale"),
        (outlier_matrix, {"shape": (3, 3), "base_std": 0.0}, "base_std"),
        (
            outlier_matrix,
            {"shape": (3, 3), "outlier_fraction": 1.5},
            "outlier_fraction",
        ),
        (outlier_matrix, {"shape": (3, 3), "outlier_scale": 0.0}, "outlier_scale"),
    ],
)
def test_invalid_parameters_raise_value_error(factory, kwargs, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory(**kwargs)
