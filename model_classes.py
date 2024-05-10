import torch.nn as nn
from torchvision import models


class SPACING(nn.Module):
    def __init__(self):
        super(SPACING, self).__init__()
        # Load the pre-trained ResNet18 model
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Get the number of features in the last layer
        num_ftrs = self.resnet.fc.in_features

        # Replace the fully connected layer with an identity pass-through
        self.resnet.fc = nn.Identity()

        # Unfreeze the last few layers of the ResNet model
        for name, param in self.resnet.named_parameters():
            if "layer4" in name or "bn" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Adding a Batch Normalization layer before Dropout
        self.batch_norm = nn.BatchNorm1d(num_ftrs)

        # Adding a Dropout layer before the final fully connected layer
        self.dropout = nn.Dropout(0.3)

        # Linear layer for binary classification or regression
        self.label_fc = nn.Linear(num_ftrs, 1)

        # Adding a ReLU activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Pass input through ResNet
        x = self.resnet(x)

        # Apply batch normalization
        x = self.batch_norm(x)

        # Apply ReLU activation
        x = self.relu(x)

        # Apply dropout
        x = self.dropout(x)

        # Final fully connected layer
        x = self.label_fc(x)

        return x
    

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 5
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.epochs_no_improve = 0
        self.early_stop = False

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.epochs_no_improve = 0

        if self.verbose and self.early_stop:
            print("Early stopping")