from train import load_my_alexnet, create_loaders, eval_fn, load_wide_resnet
from loader.dataset import MedicalImageDataset
from torch.utils.data import DataLoader
import torch
import os
from tqdm import tqdm
import pandas as pd


def create_predict_loaders(predict_transform):
    batch_size = 16
    test_image_dir = os.path.join("image_data", "test_image", "test_image")
    predict_dataset = MedicalImageDataset(label_path=None, image_dir=test_image_dir, transform=predict_transform,
                                          test=True)
    predict_loader = DataLoader(predict_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=4)
    return predict_dataset, predict_loader


def predict():
    num_classes = 3
    device = torch.device("cuda:1")
    # checkpoint_path = "trained_models/pretrained-alexnet-10_02_20-11-09-12/best.pth"
    checkpoint_path = "trained_models/pretrained-wide-resnet-10_02_20-23-32-58/best.pth"

    # model, train_transform, predict_transform = load_my_alexnet(num_classes, device)
    model, train_transform, predict_transform, _ = load_wide_resnet(num_classes, device)

    model.load_state_dict(torch.load(checkpoint_path)["model"])

    predict_dataset, predict_loader = create_predict_loaders(predict_transform)

    train_dataset, train_loader, valid_dataset, valid_loader = create_loaders(train_transform=train_transform,
                                                                              valid_transform=predict_transform,
                                                                              fold_idx=8)

    print("Evaluating model on valid dataset")
    valid_report = eval_fn(model, valid_dataset, valid_loader, device, None)
    valid_acc, valid_f1 = valid_report["accuracy"], valid_report["macro avg"]["f1-score"]
    print("Valid Acc: {}; F1: {}".format(valid_acc, valid_f1))

    predictions = [None for _ in range(len(predict_dataset))]

    for batch in tqdm(predict_loader, desc="Predicting"):
        idx_list, inputs, labels_cpu = batch["id"], batch["image"], batch["label"]
        inputs = inputs.to(device)
        output = model(inputs)

        idx_list = idx_list.numpy()
        _, preds = torch.max(output, 1)
        for idx, pred in zip(idx_list, preds.detach().cpu().numpy()):
            predictions[idx] = pred

    predict_df = pd.DataFrame()
    predict_df["ID"] = range(len(predictions))
    predict_df["Label"] = pd.Series(predictions)
    predict_df.to_csv(os.path.join("image_data", "predictions.csv"), index=False)


if __name__ == "__main__":
    predict()