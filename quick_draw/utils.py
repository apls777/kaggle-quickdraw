import numpy as np
import os


def package_dir(path=''):
    res_path = os.path.dirname(os.path.abspath(__file__))
    if path:
        res_path = os.path.join(res_path, path)

    return res_path


def project_dir(path=''):
    res_path = os.path.dirname(package_dir())
    if path:
        res_path = os.path.join(res_path, path)

    return res_path


def shuffle_arrays(arr1: np.ndarray, arr2: np.ndarray):
    assert len(arr1) == len(arr2)
    p = np.random.permutation(len(arr1))
    return arr1[p], arr2[p]
