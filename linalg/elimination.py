import numpy as np

import numpy as np


def lu(A):

    A = np.asarray(A)

    if A.shape[0] != A.shape[1]:

        raise ValueError("WRONG SHAPE")

    n = A.shape[0]

    U = np.array(A, dtype=float)

    L = np.eye(n)

    for k in range(n):

        if abs(U[k, k]) < 1e-12:
            raise ValueError(f"The pivot at {k} is zero, cannot perform LU decomposition.")
        for i in range(k+1, n):
            m = U[i, k] / U[k, k]
            U[i, :] = U[i, :] - m * U[k, :]
            L[i, k] = m
    return L, U
