import torch
from torch import nn
import torch.nn.functional as F


class TeacherNet(nn.Module):
    """
    Network architecture taken from https://github.com/pytorch/examples/blob/master/mnist/main.py
    
    98.2% accuracy after 1 epoch
    """
    def __init__(self):
        super().__init__()
    
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        return x


class StudentNet(nn.Module):
    """
    Naive linear model

    92.8% accuracy after 5 epochs, single FC layer
    """
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(28 * 28, 16)
        self.fc2 = nn.Linear(16, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x


def kd_loss(student_logits, teacher_logits, T):
    soft_targs = F.softmax(teacher_logits / T, dim=1).detach()
    soft_preds = F.log_softmax(student_logits / T, dim=1)

    distillation_loss = F.kl_div(soft_preds, soft_targs, reduction="batchmean") * T**2

    return distillation_loss