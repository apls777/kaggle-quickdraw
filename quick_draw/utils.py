import numpy as np


def shuffle_arrays(arr1: np.ndarray, arr2: np.ndarray):
    assert len(arr1) == len(arr2)
    p = np.random.permutation(len(arr1))
    return arr1[p], arr2[p]
