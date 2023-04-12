

from torch.utils.data import Dataset, default_collate
from sklearn.metrics import classification_report
from typing import List

from torch import nn
import numpy as np
import pandas as pd
import os

# TODO: This feels like library functions that oxen could provide for classification tasks
def write_predictions_to_csv(input_file, y_pred, y_indices, output_dir):
    output_file = os.path.join(output_dir, "predictions.csv")
    print(f"Writing Predictions to CSV {output_file}")
    # Write all the predictions to a dataframe so we can debug later
    # file,label,prediction,probability,is_correct
    df = pd.read_csv(input_file)
    image_paths = df['file'].tolist()
    image_labels = df['label'].tolist()
    with open(output_file, 'w') as f:
        f.write("file,label,probability,prediction,is_correct\n")
        for image_path, image_label, probability, index in zip(image_paths, image_labels, y_pred, y_indices):
            is_correct = image_label == index
            f.write(f"{image_path},{image_label},{probability[index]},{index},{is_correct}\n")

    predictions_df = pd.read_csv(output_file)
    print(predictions_df)
    
def write_metrics_to_csv(report: dict, target_names: List[str], output_dir: str):
    output_file = os.path.join(output_dir, "metrics.csv")
    print(f"Writing Metrics to CSV {output_file}")
    # Write the metrics per label to a csv we can refer to later
    metric_names = ['precision', 'recall', 'f1-score', 'support']
    with open(output_file, 'w') as f:
        f.write("label,precision,recall,f1,support\n")
        for label in report.keys():
            if not label in target_names:
                continue

            metrics = report[label]
            f.write(f"{label}")
            for metric in metric_names:
                f.write(f",{metrics[metric]}")
            f.write(f"\n")

    metrics_df = pd.read_csv(output_file)
    print(metrics_df)

def evaluate(model: nn.Module, dataset: Dataset, input_file: str, output_dir: str):
    # Collate = collect and combine
    X_test, y_true = default_collate(dataset)
    y_pred = model(X_test).detach().numpy()
    y_indices = np.argmax(y_pred, axis=1)

    # Format of results is
    # {'label 1': {'precision':0.5,
    #              'recall':1.0,
    #              'f1-score':0.67,
    #              'support':1}, ...}
    target_names = [str(i) for i in range(10)]
    report = classification_report(y_true, y_indices, target_names=target_names, output_dict=True)

    write_predictions_to_csv(input_file, y_pred, y_indices, output_dir)
    write_metrics_to_csv(report, target_names, output_dir)

