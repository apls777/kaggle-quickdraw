import csv
import json
import pickle
from quick_draw.utils import project_dir


def main():
    with open(project_dir('data/evaluation/labels.json')) as f:
        labels_map = json.load(f)

    id2label = list(range(0, len(labels_map)))
    for label, label_id in labels_map.items():
        id2label[label_id] = label

    with open(project_dir('data/kaggle_submission/3_tf_rnn/predictions.pickle'), 'rb') as f:
        submission_predictions = pickle.load(f)

    print(len(submission_predictions))

    csv_path = project_dir('data/kaggle_submission/test_simplified.csv')
    submission_csv_path = project_dir('data/kaggle_submission/3_tf_rnn/submission.csv')

    with open(submission_csv_path, 'w') as fw:
        fw.write('key_id,word\n')

        with open(csv_path) as fr:
            reader = csv.reader(fr)
            next(reader)
            i = 0
            for key_id, _, _ in reader:
                # top 3 labels
                pred_probabilities = submission_predictions[i]['logits']
                sorted_pred = sorted([(i, probability) for i, probability in enumerate(pred_probabilities)],
                                     key=lambda x: x[1], reverse=True)
                top3_labels = ' '.join([id2label[i] for i, _ in sorted_pred[:3]])

                fw.write('%s,%s\n' % (key_id, top3_labels))

                i += 1
                if i % 10000 == 0:
                    print(i)


if __name__ == '__main__':
    main()
