import sys
from quick_draw.models.estimator import train
from quick_draw.utils import import_model


def main(argv: list):
    if len(argv) < 2:
        raise ValueError('Path to configuration file is not provided')

    model_fn, input_fn, config = import_model(argv[1])

    # run training
    train(model_fn, input_fn, config)


if __name__ == '__main__':
    main(sys.argv)
