import numpy as np
import os
from scipy.special import softmax
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import torch
from code.train import load_common_transform, load_wide_resnet, load_pretrained_resnext, load_pretrained_densenet, \
    create_loaders, load_fold_models, eval_ensemble_fn
import pickle
from collections import Counter
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def ensure_path(path):
    directory = os.path.dirname(path)
    if len(directory) > 0 and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    return path


def load_from_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_to_pickle(data, path):
    with open(ensure_path(path), "wb") as f:
        pickle.dump(data, f)


def train_test_ensembles(train_features, train_labels, valid_features, valid_labels, fold, added_valid=0.):
    # concatenate both train and valid features

    if added_valid == 1:
        input_features = np.concatenate([train_features, valid_features])
        input_labels = np.concatenate([train_labels, valid_labels])
    elif added_valid > 0:
        valid_features, added_valid_features, valid_labels, added_valid_labels = \
            train_test_split(valid_features, valid_labels, test_size=added_valid, random_state=5242,
                             stratify=valid_labels)
        input_features = np.concatenate([train_features, added_valid_features])
        input_labels = np.concatenate([train_labels, added_valid_labels])
    else:
        input_features = train_features
        input_labels = train_labels

    print("Use {} to train, {} to test ensemble".format(len(input_features), len(valid_features)))

    rf_classifier = RandomForestClassifier(max_depth=4, n_estimators=15, random_state=5242)
    rf_classifier.fit(input_features, input_labels)
    rf_train_acc = rf_classifier.score(train_features, train_labels)
    print("Random forest ensemble train accuracy for fold {} = {}".format(fold, rf_train_acc))
    rf_valid_acc = rf_classifier.score(valid_features, valid_labels)
    print("Random forest ensemble valid accuracy for fold {} = {}".format(fold, rf_valid_acc))

    lr_classifier = LogisticRegression(random_state=5242)
    lr_classifier.fit(input_features, input_labels)
    lr_train_acc = lr_classifier.score(train_features, train_labels)
    print("Logistic regression train accuracy for fold {} = {}".format(fold, lr_train_acc))
    lr_valid_acc = lr_classifier.score(valid_features, valid_labels)
    print("Logistic regression valid accuracy for fold {} = {}".format(fold, lr_valid_acc))

    svm_classifier = svm.SVC(random_state=5242)
    svm_classifier.fit(input_features, input_labels)
    svm_train_acc = svm_classifier.score(train_features, train_labels)
    print("SVM train accuracy for fold {} = {}".format(fold, svm_train_acc))
    svm_valid_acc = svm_classifier.score(valid_features, valid_labels)
    print("SVM valid accuracy for fold {} = {}".format(fold, svm_valid_acc))
    return (rf_classifier, rf_valid_acc), (lr_classifier, lr_valid_acc), (svm_classifier, svm_valid_acc)


def logistic_ensemble(train_data, ensemble_method, trained_model_dir, fold_path, device=None, n_folds=10, overwrite_train_cache=False,
                      overwrite_valid_cache=False):
    print("Ensemble using {}".format(ensemble_method))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_transform, predict_transform = load_common_transform()
    all_valid_features, all_valid_labels = [], []
    classifiers = []
    model_list = []
    mv_accuracies, rf_accuracies, lr_accuracies, svm_accuracies, mv_plus_accuracies = [], [], [], [], []
    train_cache_format = "train_fold_{}_{}_cache.p"
    cache_format = "fold_{}_{}_cache.p"
    model_names = ["wide_resnet", "resnext",
                   "densenet",
                   "resnet", "inception"]

    for fold in range(0, n_folds):
        # we use the train loader to train ensemble classifier, so we don't have to use augmentation here
        train_dataset, train_loader, valid_dataset, valid_loader = create_loaders(train_data=train_data,
                                                                                  train_transform=predict_transform,
                                                                                  valid_transform=predict_transform,
                                                                                  fold_idx=fold, fold_path=fold_path,
                                                                                  shuffle_train=False)
        wide_resnet, resnext, densenet, resnet = load_fold_models(fold, device, trained_model_dir=trained_model_dir)
        wide_resnet.eval(), resnext.eval(), densenet.eval(), resnet.eval()
        print("Evaluating model on train dataset")
        all_models = [wide_resnet, resnext, densenet, resnet]
        used_model_indices = [0, 1, 2, 3]
        used_models = [all_models[i] for i in used_model_indices]
        used_model_names = [model_names[i] for i in used_model_indices]
        print("Use {}".format(used_model_names))

        train_cache_name = train_cache_format.format(fold, " ".join(used_model_names))
        if overwrite_train_cache or not os.path.exists(train_cache_name):
            train_report, train_predictions, train_labels = eval_ensemble_fn(used_models,
                                                                             train_dataset, train_loader, device,
                                                                             model_names=used_model_names)
            save_to_pickle((train_report, train_predictions, train_labels), train_cache_name)
        else:
            train_report, train_predictions, train_labels = load_from_pickle(train_cache_name)
        train_acc, train_f1 = train_report["accuracy"], train_report["macro avg"]["f1-score"]
        print("Major voting Train Acc: {}; F1: {}".format(train_acc, train_f1))
        print("Evaluating model on valid dataset for fold {}".format(fold))

        cache_name = cache_format.format(fold, " ".join(used_model_names))
        if overwrite_valid_cache or not os.path.exists(cache_name):
            valid_report, valid_predictions, valid_labels = eval_ensemble_fn(used_models,
                                                                             valid_dataset, valid_loader, device,
                                                                             model_names=used_model_names)
            save_to_pickle((valid_report, valid_predictions, valid_labels), cache_name)
        else:
            valid_report, valid_predictions, valid_labels = load_from_pickle(cache_name)
        valid_acc, valid_f1 = valid_report["accuracy"], valid_report["macro avg"]["f1-score"]
        print("Major voting Valid Acc: {}; F1: {}".format(valid_acc, valid_f1))

        # construct feature for ensembles
        # train_features = (train_size, n_models * n_classes)
        used_train_predictions = train_predictions
        used_valid_predictions = valid_predictions
        train_features = [np.concatenate([softmax(y) for y in x]) for x in zip(*used_train_predictions)]
        # valid_features = (valid_size, n_models * n_classes)
        valid_features = [np.concatenate([softmax(y) for y in x]) for x in zip(*used_valid_predictions)]
        (rf_classifier, rf_acc), (lr_classifier, lr_acc), (svm_classifier, svm_acc) = \
            train_test_ensembles(train_features, train_labels,
                                 valid_features, valid_labels, fold, added_valid=1.)
        all_valid_features.extend(valid_features)
        all_valid_labels.extend(valid_labels)

        rf_accuracies.append(rf_acc)
        mv_accuracies.append(valid_acc)
        svm_accuracies.append(svm_acc)
        lr_accuracies.append(lr_acc)

        # Calculate MV + RF method
        combined_predictions = []
        rf_predictions = rf_classifier.predict(valid_features)
        svm_predictions = svm_classifier.predict(valid_features)
        lr_predictions = lr_classifier.predict(valid_features)

        mv_predictions = [[y.argmax() for y in x] for x in zip(*used_valid_predictions)]
        for i, pred in enumerate(mv_predictions):
            cnt = Counter(pred).most_common()
            if len(cnt) > 1 and cnt[0][1] == cnt[1][1]:  # tie break
                other_pred = Counter([rf_predictions[i], svm_predictions[i], lr_predictions[i]]).most_common()[0][0]
                combined_predictions.append(other_pred)
            else:
                combined_predictions.append(cnt[0][0])
        combined_report = classification_report(valid_labels, combined_predictions, output_dict=True)
        mv_plus_accuracies.append(combined_report["accuracy"])
        print("Major voting+ Acc: {}; F1: {}".format(combined_report["accuracy"],
                                                     combined_report["macro avg"]["f1-score"]))

        if ensemble_method == "mv+":
            classifiers.append([rf_classifier, lr_classifier, svm_classifier])
        elif ensemble_method == "rf":
            classifiers.append(rf_classifier)
        elif ensemble_method == "lr":
            classifiers.append(lr_classifier)
        elif ensemble_method == "svm":
            classifiers.append(svm_classifier)
        model_list.extend(used_models)

        print("-" * 20)

    print("Major voting accuracy across folds = {}".format(np.average(mv_accuracies)))
    print("Major voting+ accuracy across folds = {}".format(np.average(mv_plus_accuracies)))
    print("RF accuracy across folds = {}".format(np.average(rf_accuracies)))
    print("LR accuracy across folds = {}".format(np.average(lr_accuracies)))
    print("SVM accuracy across folds = {}".format(np.average(svm_accuracies)))
    # all_valid_features = (valid_size * n_folds, n_models * n_classes)
    (whole_rf_classifier, whole_rf_acc), (whole_lr_classifier, whole_lr_acc), (whole_svm_classifier, whole_svm_acc) = \
        train_test_ensembles(all_valid_features, all_valid_labels, all_valid_features, all_valid_labels, "all")
    return classifiers, model_list, predict_transform, whole_rf_classifier


if __name__ == "__main__":
    logistic_ensemble()
