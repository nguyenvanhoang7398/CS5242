import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self, num_classes):
        super(SmallCNN, self).__init__()

        # (512, 512, 3)
        self.net = nn.Sequential(
            # (169, 169, 16)
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=7, stride=3),
            nn.ReLU(),
            # (84, 84, 16)
            nn.MaxPool2d(kernel_size=3, stride=2),
            # (40, 40, 64)
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=2),
            nn.ReLU(),
            # (19, 19, 64)
            nn.MaxPool2d(kernel_size=3, stride=2),
            # (17, 17, 256)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
            # (8, 8, 256)
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.classifier = nn.Sequential(
            # nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=(32 * 8 * 8), out_features=1024),
            nn.ReLU(),
            # nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=1024, out_features=num_classes),
        )
        self.init_bias()  # initialize bias

    def init_bias(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
        # original paper = 1 for Conv2d layers 2nd, 4th, and 5th conv layers

        nn.init.constant_(self.net[0].bias, 1)
        nn.init.constant_(self.net[3].bias, 1)
        nn.init.constant_(self.net[6].bias, 1)

    def forward(self, x):
        x = self.net(x)
        x = x.view(-1, 32 * 8 * 8)
        return self.classifier(x)