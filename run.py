from code.train import train_fn
from code.predict import predict_logistic_ensemble
from code.utils import create_validation_folds
import os
import sys


def run_project(train_data, test_data):
    fold_path = os.path.join("ckp", "folds")
    create_validation_folds(os.path.join(train_data, "train_label.csv"), fold_path)
    for f in range(0, 10):
        for model_n in ["wide_resnet", "densenet", "resnet", "resnext"]:
            train_fn(model_n, f, epochs=50, fold_path=fold_path)
    predict_logistic_ensemble(classifier_type="fold", fold_path=fold_path,
                              train_data=train_data, test_data=test_data)


if __name__ == "__main__":
    run_project(sys.argv[1], sys.argv[2])