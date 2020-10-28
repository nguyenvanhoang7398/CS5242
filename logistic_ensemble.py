import numpy as np
import os
from scipy.special import softmax
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import torch
from train import load_common_transform, load_wide_resnet, load_pretrained_resnext, load_pretrained_densenet, \
    create_loaders, load_fold_models, eval_ensemble_fn


def train_test_ensembles(train_features, train_labels, valid_features, valid_labels, fold):
    # concatenate both train and valid features
    input_features = np.concatenate([train_features, valid_features])
    input_labels = np.concatenate([train_labels, valid_labels])

    rf_classifier = RandomForestClassifier(max_depth=4, n_estimators=30, random_state=5242)
    rf_classifier.fit(input_features, input_labels)
    rf_acc = rf_classifier.score(valid_features, valid_labels)
    print("Random forest ensemble accuracy for fold {} = {}".format(fold, rf_acc))

    lr_classifier = LogisticRegression(random_state=5242)
    lr_classifier.fit(input_features, input_labels)
    lr_acc = lr_classifier.score(valid_features, valid_labels)
    print("Logistic regression accuracy for fold {} = {}".format(fold, lr_acc))

    svm_classifier = svm.SVC()
    svm_classifier.fit(input_features, input_labels)
    svm_acc = svm_classifier.score(valid_features, valid_labels)
    print("SVM accuracy for fold {} = {}".format(fold, svm_acc))
    print("-" * 20)
    return (rf_classifier, rf_acc), (lr_classifier, lr_acc), (svm_classifier, svm_acc)


def logistic_ensemble(n_folds=10):
    device = torch.device("cuda:3")
    train_transform, predict_transform = load_common_transform()
    all_valid_features, all_valid_labels = [], []
    classifiers = []
    model_list = []
    all_accuracies = []

    for fold in range(0, n_folds):
        # we use the train loader to train ensemble classifier, so we don't have to use augmentation here
        train_dataset, train_loader, valid_dataset, valid_loader = create_loaders(train_transform=predict_transform,
                                                                                  valid_transform=predict_transform,
                                                                                  fold_idx=fold)
        wide_resnet, resnext, densenet = load_fold_models(fold, device)
        print("Evaluating model on train dataset")
        train_report, train_predictions, train_labels = eval_ensemble_fn([wide_resnet, resnext, densenet],
                                                                         train_dataset, train_loader, device)
        train_acc, train_f1 = train_report["accuracy"], train_report["macro avg"]["f1-score"]
        print("Train Acc: {}; F1: {}".format(train_acc, train_f1))
        print("Evaluating model on valid dataset")
        valid_report, valid_predictions, valid_labels = eval_ensemble_fn([wide_resnet, resnext, densenet],
                                                                         valid_dataset, valid_loader, device)
        valid_acc, valid_f1 = valid_report["accuracy"], valid_report["macro avg"]["f1-score"]
        print("Valid Acc: {}; F1: {}".format(valid_acc, valid_f1))

        # construct feature for ensembles
        train_features = [np.concatenate([softmax(y) for y in x]) for x in zip(*train_predictions)]
        valid_features = [np.concatenate([softmax(y) for y in x]) for x in zip(*valid_predictions)]
        (rf_classifier, rf_acc), (lr_classifier, lr_acc), (svm_classifier, svm_acc) = \
            train_test_ensembles(train_features, train_labels,
                                 valid_features, valid_labels, fold)
        all_valid_features.extend(valid_features)
        all_valid_labels.extend(valid_labels)
        all_accuracies.append(rf_acc)

        classifiers.append(rf_classifier)
        model_list.extend([wide_resnet, resnext, densenet])

    print("Ensemble accuracy across folds: {}".format(np.average(all_accuracies)))
    train_test_ensembles(all_valid_features, all_valid_labels, all_valid_features, all_valid_labels, "all")
    return classifiers, model_list, predict_transform


if __name__ == "__main__":
    logistic_ensemble()