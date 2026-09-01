import numpy as np
import pytest
from linalg.elimination import lu


def test_lu_reconstructs_a():
    A = np.array([[1., 3., 0.], [2., 4., 0.], [2., 0., 1.]])
    L, U = lu(A)
    assert np.allclose(L @ U, A)