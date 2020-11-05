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
from logistic_ensemble import logistic_ensemble, save_to_pickle, load_from_pickle
from scipy.special import softmax


def create_predict_loaders(predict_transform):
    batch_size = 16
    test_image_dir = os.path.join("test_image", "test_image")
    predict_dataset = MedicalImageDataset(label_path=None, image_dir=test_image_dir, transform=predict_transform,
                                          test=True)
    predict_loader = DataLoader(predict_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=4)
    return predict_dataset, predict_loader


def predict():
    train_transform, predict_transform = load_common_transform()
    device = torch.device("cuda")

    all_models = []

    for fold in range(10):
        train_dataset, train_loader, valid_dataset, valid_loader = create_loaders(train_transform=train_transform,
                                                                                  valid_transform=predict_transform,
                                                                                  fold_idx=fold,
                                                                                  fold_name="folds",
                                                                                  shuffle_train=False)
        wide_resnet, resnext, densenet, resnet, inception = load_fold_models(fold, device,
                                                                             trained_model_dir="trained_models_0.97945")
        print("-" * 10)
        print("Fold: {}".format(fold))
        print("Evaluating model on train dataset of fold")
        train_report, _, _ = eval_ensemble_fn([wide_resnet, resnext, densenet, resnet, inception], train_dataset, train_loader, device)
        train_acc, train_f1 = train_report["accuracy"], train_report["macro avg"]["f1-score"]
        print("Train Acc: {}; F1: {}".format(train_acc, train_f1))

        print("Evaluating model on valid dataset")
        valid_report, _, _ = eval_ensemble_fn([wide_resnet, resnext, densenet, resnet, inception], valid_dataset, valid_loader, device)
        valid_acc, valid_f1 = valid_report["accuracy"], valid_report["macro avg"]["f1-score"]
        print("Valid Acc: {}; F1: {}".format(valid_acc, valid_f1))
        all_models.extend([wide_resnet, resnext, densenet, resnet, inception])

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
    predict_df.to_csv(os.path.join(outfile), index=False)


def predict_all():
    outfile = "predictions.csv"
    trained_model_dir = "trained_models_recover"
    device = torch.device("cuda")
    overwrite_train_cache = True

    train_transform, predict_transform = load_common_transform()
    cache_format = "fold_{}_{}_cache.p"
    model_names = ["wide_resnet", "resnext",
                   "densenet",
                   "resnet"]

    train_dataset, train_loader, valid_dataset, valid_loader = create_loaders(train_transform=predict_transform,
                                                                              valid_transform=predict_transform,
                                                                              fold_idx="all", fold_name="folds",
                                                                              shuffle_train=False)
    wide_resnet, resnext, densenet, resnet = load_fold_models("all", device, trained_model_dir=trained_model_dir)

    wide_resnet.eval(), resnext.eval(), densenet.eval(), resnet.eval()
    print("Evaluating model on train dataset")
    all_models = [wide_resnet, resnext, densenet, resnet]
    used_model_indices = [0, 1, 2, 3]
    print("Use {}".format([model_names[i] for i in used_model_indices]))
    model_list = [all_models[i] for i in used_model_indices]

    train_cache_name = cache_format.format("all", "_".join(model_names))
    if overwrite_train_cache or not os.path.exists(train_cache_name):
        train_report, train_predictions, train_labels = eval_ensemble_fn(model_list,
                                                                         train_dataset, train_loader, device,
                                                                         model_names=model_names)
        save_to_pickle((train_report, train_predictions, train_labels), train_cache_name)
    else:
        train_report, train_predictions, train_labels = load_from_pickle(train_cache_name)
    train_acc, train_f1 = train_report["accuracy"], train_report["macro avg"]["f1-score"]
    print("Major voting Train Acc: {}; F1: {}".format(train_acc, train_f1))

    n_folds = 1
    ensemble_method = "mv"
    classifiers = []
    ensemble_plus_models = ""
    final_predictions = predict_fn(device, predict_transform, model_list, n_folds, ensemble_method, classifiers,
                                   ensemble_plus_models, "", None)

    predict_df = pd.DataFrame()
    predict_df["ID"] = range(len(final_predictions))
    predict_df["Label"] = pd.Series(final_predictions)
    predict_df.to_csv(os.path.join(outfile), index=False)


def predict_logistic_ensemble(classifier_type="fold"):
    outfile = "predictions.csv"
    ensemble_method = "mv"
    ensemble_plus_models = ["rf"]
    trained_model_dir = "trained_models"
    device = torch.device("cuda")
    n_folds = 10
    classifiers, model_list, predict_transform, whole_classifier = logistic_ensemble(ensemble_method=ensemble_method,
                                                                                     device=device, n_folds=n_folds,
                                                                                     overwrite_train_cache=False,
                                                                                     overwrite_valid_cache=False,
                                                                                     trained_model_dir=trained_model_dir)

    final_predictions = predict_fn(device, predict_transform, model_list, n_folds, ensemble_method, classifiers,
                                   ensemble_plus_models, classifier_type, whole_classifier)

    predict_df = pd.DataFrame()
    predict_df["ID"] = range(len(final_predictions))
    predict_df["Label"] = pd.Series(final_predictions)
    predict_df.to_csv(os.path.join(outfile), index=False)


def predict_fn(device, predict_transform, model_list, n_folds, ensemble_method, classifiers,
               ensemble_plus_models, classifier_type, whole_classifier):
    n_models = int(len(model_list) / n_folds)
    predict_dataset, predict_loader = create_predict_loaders(predict_transform)
    all_numeric_predictions = [[] for _ in range(len(model_list))]
    final_predictions = []

    for batch in tqdm(predict_loader, desc="Predicting"):
        idx_list, inputs, labels_cpu = batch["id"], batch["image"], batch["label"]
        inputs = inputs.to(device)
        for model_idx, model in enumerate(model_list):
            output = model(inputs)

            all_numeric_predictions[model_idx].extend(output.detach().cpu().numpy())

    # all_numeric_predictions = (n_models * n_folds, n_folds * valid_size, n_classes)
    for numeric_predictions in zip(*all_numeric_predictions):
        fold_predictions = []
        for fold_idx in range(n_folds):
            # fold_numeric_predictions = (n_models)
            fold_numeric_predictions = numeric_predictions[n_models * fold_idx: n_models * (fold_idx + 1)]

            if ensemble_method == "mv":
                # Major voting ensemble
                model_predictions = [x.argmax() for x in fold_numeric_predictions]
                prediction = Counter(model_predictions).most_common()[0][0]
            elif ensemble_method == "mv+":
                # Major voting ensemble
                model_predictions = [x.argmax() for x in fold_numeric_predictions]
                cnt = Counter(model_predictions).most_common()
                if len(cnt) > 1 and cnt[0][1] == cnt[1][1]:  # tie break
                    feature = np.concatenate([softmax(x) for x in fold_numeric_predictions])
                    rf_classifier, lr_classifier, svm_classifier = classifiers[fold_idx]
                    predictions = []
                    if "rf" in ensemble_plus_models:
                        predictions.append(rf_classifier.predict(feature.reshape(1, -1))[0])
                    if "lr" in ensemble_plus_models:
                        predictions.append(lr_classifier.predict(feature.reshape(1, -1))[0])
                    if "svm" in ensemble_plus_models:
                        predictions.append(svm_classifier.predict(feature.reshape(1, -1))[0])
                    # major voting on other ensemble method
                    prediction = Counter(predictions).most_common()[0][0]
                else:
                    prediction = cnt[0][0]
            else:
                feature = np.concatenate([softmax(x) for x in fold_numeric_predictions])
                # load the correct classifier from the specific fold
                fold_classifier = classifiers[fold_idx] if classifier_type == "fold" else whole_classifier
                prediction = fold_classifier.predict(feature.reshape(1, -1))[0]
            fold_predictions.append(prediction)
        final_prediction = Counter(fold_predictions).most_common()[0][0]
        final_predictions.append(final_prediction)

    return final_predictions


if __name__ == "__main__":
    # predict()
    predict_logistic_ensemble(classifier_type="fold")
    # predict_all()