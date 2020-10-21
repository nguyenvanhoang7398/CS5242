from train import create_loaders, eval_fn, eval_ensemble_fn, load_wide_resnet, load_pretrained_densenet, \
    load_pretrained_resnext, load_common_transform, load_fold_models
from loader.dataset import MedicalImageDataset
from torch.utils.data import DataLoader
import torch
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from collections import Counter
from logistic_ensemble import logistic_ensemble
from scipy.special import softmax


def create_predict_loaders(predict_transform):
    batch_size = 16
    test_image_dir = os.path.join("image_data", "test_image", "test_image")
    predict_dataset = MedicalImageDataset(label_path=None, image_dir=test_image_dir, transform=predict_transform,
                                          test=True)
    predict_loader = DataLoader(predict_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=4)
    return predict_dataset, predict_loader


def predict():
    train_transform, predict_transform = load_common_transform()
    device = torch.device("cuda:3")

    all_models = []

    for fold in range(10):
        train_dataset, train_loader, valid_dataset, valid_loader = create_loaders(train_transform=train_transform,
                                                                                  valid_transform=predict_transform,
                                                                                  fold_idx=fold)
        wide_resnet, resnext, densenet = load_fold_models(fold, device)
        print("-" * 10)
        print("Fold: {}".format(fold))
        print("Evaluating model on train dataset of fold")
        train_report, _, _ = eval_ensemble_fn([wide_resnet, resnext, densenet], train_dataset, train_loader, device)
        train_acc, train_f1 = train_report["accuracy"], train_report["macro avg"]["f1-score"]
        print("Train Acc: {}; F1: {}".format(train_acc, train_f1))

        print("Evaluating model on valid dataset")
        valid_report, _, _ = eval_ensemble_fn([wide_resnet, resnext, densenet], valid_dataset, valid_loader, device)
        valid_acc, valid_f1 = valid_report["accuracy"], valid_report["macro avg"]["f1-score"]
        print("Valid Acc: {}; F1: {}".format(valid_acc, valid_f1))
        all_models.extend([wide_resnet, resnext, densenet])

    predict_dataset, predict_loader = create_predict_loaders(predict_transform)
    predict_ensemble_fn(all_models, predict_loader, device)


def predict_ensemble_fn(model_list, predict_loader, device, outfile="predictions.csv"):
    predictions = [[] for _ in range(len(model_list))]

    for batch in tqdm(predict_loader, desc="Predicting"):
        for model_idx, model in enumerate(model_list):
            idx_list, inputs, labels_cpu = batch["id"], batch["image"], batch["label"]
            inputs = inputs.to(device)
            output = model(inputs)

            _, preds = torch.max(output, 1)
            predictions[model_idx].extend(preds.detach().cpu().numpy())

    # combine eval prediction by major voting
    final_predictions = []
    for x in zip(*predictions):
        final_predictions.append(Counter(x).most_common()[0][0])

    predict_df = pd.DataFrame()
    predict_df["ID"] = range(len(final_predictions))
    predict_df["Label"] = pd.Series(final_predictions)
    predict_df.to_csv(os.path.join("image_data", outfile), index=False)


def predict_logistic_ensemble():
    outfile = "predictions.csv"
    device = torch.device("cuda:3")
    n_folds, n_models = 10, 3
    classifiers, model_list, predict_transform = logistic_ensemble(n_folds=n_folds)
    predict_dataset, predict_loader = create_predict_loaders(predict_transform)
    all_numeric_predictions = [[] for _ in range(len(model_list))]
    final_predictions = []

    for batch in tqdm(predict_loader, desc="Predicting"):
        for model_idx, model in enumerate(model_list):
            idx_list, inputs, labels_cpu = batch["id"], batch["image"], batch["label"]
            inputs = inputs.to(device)
            output = model(inputs)

            all_numeric_predictions[model_idx].extend(output.detach().cpu().numpy())

    for numeric_predictions in zip(*all_numeric_predictions):
        fold_predictions = []
        for fold_idx in range(n_folds):
            fold_numeric_predictions = numeric_predictions[n_models * fold_idx: n_models * (fold_idx+1)]
            feature = np.concatenate([softmax(x) for x in fold_numeric_predictions])
            fold_classifier = classifiers[fold_idx]
            prediction = fold_classifier.predict(feature.reshape(1, -1))[0]
            fold_predictions.append(prediction)
        final_prediction = Counter(fold_predictions).most_common()[0][0]
        final_predictions.append(final_prediction)

    predict_df = pd.DataFrame()
    predict_df["ID"] = range(len(final_predictions))
    predict_df["Label"] = pd.Series(final_predictions)
    predict_df.to_csv(os.path.join("image_data", outfile), index=False)


if __name__ == "__main__":
    # predict()
    predict_logistic_ensemble()