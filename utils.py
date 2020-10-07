import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import os
import numpy as np
from datetime import datetime


def get_exp_name(model_name):
    return "{}-{}".format(model_name, datetime.now().strftime("%D-%H-%M-%S").replace("/", "_"))


def create_validation_folds(train_label_path, output, n_folds=10):
    train_labels = pd.read_csv(train_label_path)
    X = np.array(train_labels["ID"].tolist())
    y = np.array(train_labels["Label"].tolist())

    skf = StratifiedKFold(n_splits=n_folds)
    for fold_idx, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        fold_folder = os.path.join(output, "fold_{}".format(fold_idx))
        if not os.path.exists(fold_folder):
            os.makedirs(fold_folder)

        train_path = os.path.join(fold_folder, "train_labels.csv")
        valid_path = os.path.join(fold_folder, "valid_labels.csv")

        train_df = pd.DataFrame()
        train_df["ID"] = pd.Series(X_train)
        train_df["Label"] = y_train

        valid_df = pd.DataFrame()
        valid_df["ID"] = pd.Series(X_test)
        valid_df["Label"] = y_test

        train_df.to_csv(train_path, index=False)
        valid_df.to_csv(valid_path, index=False)


def compare_2_submission(base, new):
    base_df = pd.read_csv(base)
    new_df = pd.read_csv(new)

    base_y = np.array(base_df["Label"].tolist())
    new_y = np.array(new_df["Label"].tolist())

    print("Accuracy = {}".format(accuracy_score(base_y, new_y)))


if __name__ == "__main__":
    # create_validation_folds(os.path.join("image_data", "train_label.csv"), os.path.join("image_data", "folds"))
    compare_2_submission("prediction_wide_resnet.csv", "prediction_wide_resnet_tuned.csv")
    compare_2_submission("model-preds.csv", "prediction_wide_resnet.csv")