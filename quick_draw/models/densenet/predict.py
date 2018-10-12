import os
import pickle
from quick_draw.models.estimator import predict
from quick_draw.utils import project_dir


if __name__ == '__main__':
    output_path = project_dir(os.path.join('notebooks', 'data', 'predictions.pickle'))
    labels_map_path = project_dir(os.path.join('notebooks', 'data', 'labels.json'))
    predictor = predict('densenet', labels_map_path, [project_dir('notebooks/data/file_1.tfrecords')])

    predictions = []
    for i, prediction in enumerate(predictor):
        if i % 100 == 0:
            print(i)

        predictions.append(prediction)

    with open(output_path, 'wb') as f:
        pickle.dump(predictions, f)
