# October 4th

## Completed tasks

* ~~Implement DenseNet + Pretrained~~
* ~~Implement Gaussian Noise Injection~~
* ~~Unfreeze last layers~~
* ~~Add L2 regularization~~

## Experiments

### New experiment
| Model | Fold | Accuracy | F1 score |
| ----- | ----- | ----- | -----
| Wide Resnet (unfreeze last 2 layers) | 0 | 0.9829 | 0.9828|
| | 1 | 0.9915 | 0.9915 |
| | 2 | 0.9316 | 0.9332 |
| | 3 | 0.9573 | 0.9567 |
| | 4 | 0.9482 | 0.9476 |
| | 5 | 0.9483 | 0.9500 |
| | 6 | 0.9224 | 0.9208 |
| | 7 | 0.9397 | 0.9431 |
| | 8 | 0.9483 | 0.9496 |
| | 9 | 0.9655 | 0.9669 |
| | **Average** | **0.9536** | **0.9542**

## Submissions

| Model | Score |
| ----- | ----- |
| Wide Resnet (unfreeze last 2 layers) | 0.96575 |
