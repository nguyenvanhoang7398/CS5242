from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report
from loader.dataset import MedicalImageDataset
import os
from torchvision import transforms
from torchvision import models
from torchvision.transforms import functional
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from collections import Counter
import time

import torch
from models.alexnet import AlexNet
from loader.augmentor import AddGaussianNoise
import torch.optim as optim
from utils import get_exp_name


def load_transform(image_size=256, crop_size=224):
    return load_common_transform(image_size, crop_size)


def load_common_transform(image_size=256, crop_size=224):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=(-5, 5)),
        transforms.Resize(image_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        # AddGaussianNoise(0., 0.005)
    ])

    valid_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return train_transform, valid_transform


def load_old_transform(image_size=256, crop_size=224):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=(-15, 15)),
        transforms.Resize(image_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        # AddGaussianNoise(0., 0.05)
    ])

    valid_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return train_transform, valid_transform


def load_wide_resnet(num_classes, device):
    model = models.wide_resnet101_2(pretrained=True).to(device)
    trained_layer_indices = [7, 9]

    for i, child in enumerate(model.children()):
        print(i, child)
        if i not in trained_layer_indices:
            for param in child.parameters():
                param.requires_grad = False
    num_ftrs = model.fc.in_features
    # model.fc = nn.Sequential(nn.Linear(num_ftrs, 256), nn.Linear(256, num_classes)).to(device)
    model.fc = nn.Linear(num_ftrs, num_classes).to(device)

    train_transform, valid_transform = load_transform(image_size=256, crop_size=224)
    return model, train_transform, valid_transform, "pretrained-wide-resnet"


def load_pretrained_densenet(num_classes, device):
    model = models.densenet201(pretrained=True).to(device)
    trained_layer_indices = [10, 11]

    for i, child in enumerate(model.features.children()):
        print(i, child)
        if i not in trained_layer_indices:
            for param in child.parameters():
                param.requires_grad = False
    num_ftrs = model.classifier.in_features
    model.fc = nn.Linear(num_ftrs, num_classes).to(device)

    train_transform, valid_transform = load_transform(image_size=256, crop_size=224)
    return model, train_transform, valid_transform, "pretrained-densenet"


def load_pretrained_resnext(num_classes, device):
    model = models.resnext101_32x8d(pretrained=True).to(device)
    trained_layer_indices = [7, 9]

    for i, child in enumerate(model.children()):
        if i not in trained_layer_indices:
            for param in child.parameters():
                param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes).to(device)

    train_transform, valid_transform = load_transform(image_size=256, crop_size=224)
    return model, train_transform, valid_transform, "pretrained-resnext"


def create_loaders(train_transform, valid_transform, fold_idx=0):
    batch_size = 64
    train_label_path = os.path.join("image_data", "folds", "fold_{}".format(fold_idx), "train_labels.csv")
    valid_label_path = os.path.join("image_data", "folds", "fold_{}".format(fold_idx), "valid_labels.csv")
    image_dir = os.path.join("image_data", "train_image", "train_image")

    train_dataset = MedicalImageDataset(label_path=train_label_path, image_dir=image_dir, transform=train_transform)
    valid_dataset = MedicalImageDataset(label_path=valid_label_path, image_dir=image_dir, transform=valid_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=4)
    return train_dataset, train_loader, valid_dataset, valid_loader


def train_fn(model_name, fold_idx):
    print("Training {} on fold {}".format(model_name, fold_idx))
    num_classes = 3
    epochs = 100

    device = torch.device("cuda:3")
    # device = torch.device("cpu")

    if model_name == "wide_resnet":
        model, train_transform, valid_transform, model_name = load_wide_resnet(num_classes, device)
    elif model_name == "densenet":
        model, train_transform, valid_transform, model_name = load_pretrained_densenet(num_classes, device)
    elif model_name == "resnext":
        model, train_transform, valid_transform, model_name = load_pretrained_resnext(num_classes, device)
    else:
        raise ValueError("Unsupported model {}".format(model_name))

    exp_name = get_exp_name(model_name, fold_idx)
    writer = SummaryWriter(os.path.join("runs", exp_name))

    train_dataset, train_loader, valid_dataset, valid_loader = create_loaders(train_transform, valid_transform,
                                                                              fold_idx=fold_idx)
    train_size = len(train_dataset)

    optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001, weight_decay=1e-3)
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_acc_f1 = 0.

    for epoch in range(epochs):

        train_predictions, train_labels = [], []
        epoch_loss = 0.

        for batch in tqdm(train_loader, desc="Training"):
            inputs, labels_cpu = batch["image"], batch["label"]
            inputs = inputs.to(device)
            labels = labels_cpu.to(device)

            output = model(inputs)
            loss = F.cross_entropy(output, labels)
            epoch_loss += loss.item()

            # update the parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, preds = torch.max(output, 1)
            train_predictions.extend(preds.detach().cpu().numpy())
            train_labels.extend(labels_cpu.numpy())

        # lr_scheduler.step()

        print("Finish training Epoch: {}; Loss: {}".format(epoch+1, epoch_loss / train_size))
        train_report = classification_report(train_labels, train_predictions, output_dict=True)
        train_acc, train_f1 = train_report["accuracy"], train_report["macro avg"]["f1-score"]
        print("Train Acc: {}; F1: {}".format(train_acc, train_f1))
        for metric_name, metric_values in train_report.items():
            if type(metric_values) == dict:
                writer.add_scalars(main_tag="train_{}".format(metric_name), tag_scalar_dict=metric_values,
                                   global_step=epoch)
            else:
                writer.add_scalar(tag="train_{}".format(metric_name), scalar_value=metric_values,
                                  global_step=epoch)
        print("Begin evaluating")
        valid_report = eval_fn(model, valid_dataset, valid_loader, device, 1)
        valid_acc, valid_f1 = valid_report["accuracy"], valid_report["macro avg"]["f1-score"]
        print("Valid Acc: {}; F1: {}".format(valid_acc, valid_f1))

        for metric_name, metric_values in valid_report.items():
            if type(metric_values) == dict:
                writer.add_scalars(main_tag="valid_{}".format(metric_name), tag_scalar_dict=metric_values,
                                   global_step=epoch)
            else:
                writer.add_scalar(tag="valid_{}".format(metric_name), scalar_value=metric_values,
                                  global_step=epoch)

        avg_acc_f1 = (valid_acc + valid_f1) / 2
        if avg_acc_f1 > best_acc_f1:
            best_acc_f1 = avg_acc_f1
            checkpoint_dir = os.path.join("trained_models", exp_name)
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, "best.pth")
            state = {
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict(),
            }
            torch.save(state, checkpoint_path)
            print("Save best model to {}".format(checkpoint_path))


def eval_ensemble_fn(model_list, valid_dataset, valid_loader, device):
    for model in model_list:
        model.eval()

    all_eval_predictions, eval_labels = [[] for _ in range(len(model_list))], []
    eval_loss = 0.

    with torch.no_grad():
        for model_idx, model in enumerate(model_list):
            for batch in tqdm(valid_loader, desc="Evaluating"):
                inputs, labels_cpu = batch["image"], batch["label"]
                inputs = inputs.to(device)
                labels = labels_cpu.to(device)

                output = model(inputs)
                loss = F.cross_entropy(output, labels)
                eval_loss += loss.item()

                _, preds = torch.max(output, 1)
                all_eval_predictions[model_idx].extend(preds.detach().cpu().numpy())

                if model_idx == 0:
                    eval_labels.extend(labels_cpu.numpy())

    # combine eval prediction by major voting
    eval_predictions = []
    for x in zip(*all_eval_predictions):
        eval_predictions.append(Counter(x).most_common()[0][0])

    print("Finish evaluating, Loss: {}".format(eval_loss / len(valid_dataset)))
    report = classification_report(eval_labels, eval_predictions, output_dict=True)
    return report


def eval_fn(model, valid_dataset, valid_loader, device, eval_times=5):
    model.eval()

    all_eval_predictions, eval_labels = [[] for _ in range(eval_times)], []
    eval_loss = 0.

    with torch.no_grad():
        for eval_time in range(eval_times):
            for batch in tqdm(valid_loader, desc="Evaluating"):
                inputs, labels_cpu = batch["image"], batch["label"]
                inputs = inputs.to(device)
                labels = labels_cpu.to(device)

                output = model(inputs)
                loss = F.cross_entropy(output, labels)
                eval_loss += loss.item()

                _, preds = torch.max(output, 1)
                all_eval_predictions[eval_time].extend(preds.detach().cpu().numpy())

                if eval_time == 0:
                    eval_labels.extend(labels_cpu.numpy())

    # combine eval prediction by major voting
    eval_predictions = []
    for x in zip(*all_eval_predictions):
        eval_predictions.append(Counter(x).most_common()[0][0])

    print("Finish evaluating, Loss: {}".format(eval_loss / len(valid_dataset)))
    report = classification_report(eval_labels, eval_predictions, output_dict=True)
    return report


if __name__ == "__main__":
    start_time = time.time()
    for fold in range(6, 10):
        for model_n in ["wide_resnet", "densenet", "resnext"]:
            train_fn(model_n, fold)
    end_time = time.time()
    print("Finished in {}".format(end_time - start_time))