import json
import os
from quick_draw.utils import package_dir


def load_model_params(model_name):
    filename = 'params.json'
    if 'QUICK_DRAW_ENV' in os.environ:
        filename = 'params.%s.json' % os.environ['QUICK_DRAW_ENV']

    with open(package_dir(os.path.join('models', model_name, filename))) as f:
        res = json.load(f)

    return res
