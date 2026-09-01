import numpy as np
import pytest
from linalg.elimination import lu, plu


def test_lu_reconstructs_a():
    A = np.array([[1., 3., 0.], [2., 4., 0.], [2., 0., 1.]])
    L, U = lu(A)
    assert np.allclose(L @ U, A)

def test_u_is_upper_triangular():
    A = np.array([[1., 3., 0.], [2., 4., 0.], [2., 0., 1.]])
    L, U = lu(A)
    assert np.allclose(U, np.triu(U))

def test_l_is_unit_lower_triangular():
    A = np.array([[1., 3., 0.], [2., 4., 0.], [2., 0., 1.]])
    L, U = lu(A)
    assert np.allclose(L, np.tril(L))
    assert np.allclose(1, np.diag(L))

def test_non_square_raises():
    A = np.array([[1., 3., 0.], [2., 4., 0.]])
    with pytest.raises(ValueError):
        lu(A)

def test_zero_pivots():
    A = np.array([[0., 3., 1.], [2., 4., 0.], [2., 0., 1.]])
    with pytest.raises(ValueError):
        lu(A)

def test_different_A_sizes():
    rng = np.random.default_rng(0)

    for n in range(3, 9):
        A = rng.normal(size=(n, n))
        L, U = lu(A)
        assert np.allclose(L @ U, A)

def test_A_unchanged():
    A = np.array([[1., 3., 0.], [2., 4., 0.], [2., 0., 1.]])
    A_before = A.copy()
    lu(A)
    assert np.array_equal(A, A_before)

def test_correct_type():
    A = np.array([[2, 1], [6, 4]])
    L, U = lu(A)
    assert U.dtype == float
    assert np.allclose(L @ U, A)





def test_lu_reconstructs_ap():
    A = np.array([[1., 3., 0.], [2., 4., 0.], [2., 0., 1.]])
    L, U, P = plu(A)
    assert np.allclose(L @ U, P @ A)

def test_u_is_upper_triangularp():
    A = np.array([[1., 3., 0.], [2., 4., 0.], [2., 0., 1.]])
    L, U, P = plu(A)
    assert np.allclose(U, np.triu(U))

def test_l_is_unit_lower_triangularp():
    A = np.array([[1., 3., 0.], [2., 4., 0.], [2., 0., 1.]])
    L, U, P = plu(A)
    assert np.allclose(L, np.tril(L))
    assert np.allclose(1, np.diag(L))

def test_non_square_raisesp():
    A = np.array([[1., 3., 0.], [2., 4., 0.]])
    with pytest.raises(ValueError):
        plu(A)

def test_non_singular_raisesp():
    A = np.array([[1., 2], [2., 4]])
    with pytest.raises(ValueError):
        plu(A)

def test_zero_pivotsp():
    A = np.array([[0., 3., 1.], [2., 4., 0.], [2., 0., 1.]])
    L, U, P = plu(A)
    assert np.allclose(L @ U, P @ A)

def test_different_A_sizesp():
    rng = np.random.default_rng(0)

    for n in range(3, 9):
        A = rng.normal(size=(n, n))
        L, U, P = plu(A)
        assert np.allclose(L @ U, P @ A)

def test_A_unchangedp():
    A = np.array([[1., 3., 0.], [2., 4., 0.], [2., 0., 1.]])
    A_before = A.copy()
    plu(A)
    assert np.array_equal(A, A_before)

def test_correct_typep():
    A = np.array([[2, 1], [6, 4]])
    L, U, P = plu(A)
    assert U.dtype == float
    assert np.allclose(L @ U, P @ A)

def test_P_check():
    A = np.array([[1., 3., 0.], [2., 4., 0.], [2., 0., 1.]])
    L, U, P = plu(A)
    assert np.allclose(P.sum(axis = 0), 1)
    assert np.allclose(P.sum(axis = 1), 1) 
    assert np.all((P == 0) | (P == 1))