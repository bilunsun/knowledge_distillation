import argparse
import os
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy


from data_utils import get_mnist_train_test
from neural_network_utils import StudentNet, TeacherNet, kd_loss


class KnowledgeDistillation(pl.LightningModule):
    def __init__(self, student, teacher, train_loader, test_loader, T=1, alpha=1):
        super().__init__()

        self.teacher = teacher
        self.student = student
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.T = T
        self.alpha = alpha

    def forward(self, x):
        student_logits = self.student(x)
        teacher_logits = self.teacher(x)

        return student_logits, teacher_logits

    def training_step(self, batch, batch_index):
        x, y = batch

        student_logits, teacher_logits = self(x)
        loss = kd_loss(student_logits, teacher_logits, y, self.T, self.alpha)

        return loss
    
    def validation_step(self, batch, batch_index):
        x, y = batch
        student_logits, _ = self(x)
        loss = F.cross_entropy(student_logits, y)

        preds = torch.argmax(student_logits, dim=1)
        acc = accuracy(preds, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return optim.Adam(self.student.parameters(), lr=1e-3)
    
    def train_dataloader(self):
        return self.train_loader
    
    def val_dataloader(self):
        return self.test_loader


def main():
    train_loader, test_loader = get_mnist_train_test()

    teacher = TeacherNet()
    teacher.load_state_dict(torch.load(".models/teacher.pth"))

    student = StudentNet()

    kd = KnowledgeDistillation(student, teacher, train_loader, test_loader)
    trainer = pl.Trainer(max_epochs=1, gpus=1)
    trainer.fit(kd)


if __name__ == "__main__":
    main()