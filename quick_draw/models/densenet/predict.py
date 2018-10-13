import pickle
from quick_draw.models.estimator import predict
from quick_draw.utils import project_dir


if __name__ == '__main__':
    labels_map_path = project_dir('data/evaluation/labels.json')
    # file_path = project_dir('data/evaluation/file_1.tfrecords')
    # output_path = project_dir('data/evaluation/predictions.pickle')
    file_path = project_dir('data/submission/test_simplified.tfrecords')
    output_path = project_dir('data/submission/predictions.pickle')

    predictor = predict('densenet', labels_map_path, [file_path])

    predictions = []
    for i, prediction in enumerate(predictor):
        if i % 100 == 0:
            print(i)

        predictions.append(prediction)

    with open(output_path, 'wb') as f:
        pickle.dump(predictions, f)
