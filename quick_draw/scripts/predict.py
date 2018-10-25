import sys
import pickle
from quick_draw.models.estimator import predict
from quick_draw.utils import project_dir
from quick_draw.utils import import_model


def main(argv: list):
    if len(argv) < 2:
        raise ValueError('Path to configuration file is not provided')

    config_path = argv[1]
    file_path = argv[2]
    output_path = project_dir(argv[3])

    model_fn, input_fn, config = import_model(config_path)

    # file_path = 's3://quickdraw-datasets-us-east-2/stroke3/file_1.tfrecords'
    # output_path = project_dir('data/eval_1_tf_rnn.pickle')

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
