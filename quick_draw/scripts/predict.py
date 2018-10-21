import sys
import pickle
from quick_draw.models.estimator import predict
from quick_draw.utils import project_dir
from quick_draw.utils import import_model


def main(argv: list):
    if len(argv) < 2:
        raise ValueError('Path to configuration file is not provided')

    model_fn, input_fn, config = import_model(argv[1])

    # file_path = project_dir('data/evaluation/file_1.tfrecords')
    # output_path = project_dir('data/evaluation/predictions.pickle')
    file_path = project_dir('data/submission/test_simplified.tfrecords')
    output_path = project_dir('data/submission/predictions.pickle')

    predictor = predict(model_fn, input_fn, config, [file_path])

    predictions = []
    for i, prediction in enumerate(predictor):
        if i % 100 == 0:
            print(i)

        predictions.append(prediction)

    with open(output_path, 'wb') as f:
        pickle.dump(predictions, f)


if __name__ == '__main__':
    main(sys.argv)
