import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ToyNet(nn.Module):
    def __init__(self):
        super(ToyNet, self).__init__()
        self.features = nn.Sequential(
            # # define the extracting network here
            nn.Linear(2, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            # define the classifier network here
            nn.Linear(10, 2),
        )

    def forward(self, x):
        # define the forward function here
        x = self.features(x)
        x = self.classifier(x)
        return x
