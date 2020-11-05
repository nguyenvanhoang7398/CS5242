from train import train_fn
from predict import predict_logistic_ensemble
from utils import create_validation_folds
import os


def run_project():
    create_validation_folds(os.path.join("train_label.csv"), os.path.join("folds"))
    for f in range(0, 10):
        for model_n in ["wide_resnet", "densenet", "resnet", "resnext"]:
            train_fn(model_n, f, epochs=50)
    predict_logistic_ensemble(classifier_type="fold")


if __name__ == "__main__":
    run_project()