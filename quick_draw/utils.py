import json
from importlib import import_module
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


def read_json(file_path):
    with open(file_path) as f:
        res = json.load(f)

    return res


def import_model(config_path):
    # read model configuration
    config = read_json(project_dir(config_path))

    # model function for the Estimator
    model_fn = getattr(import_module('quick_draw.models.%s.model' % config['model_package']), 'model_fn')

    # input function for the Estimator
    input_fn = getattr(import_module('quick_draw.models.%s.input' % config['model_package']), 'input_fn')

    return model_fn, input_fn, config
