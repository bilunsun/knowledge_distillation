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
    def __init__(self, student, teacher, train_loader, test_loader, T=3, alpha=1):
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
        loss = kd_loss(student_logits, teacher_logits, self.T)

        self.log("student_train_loss", loss)

        return loss
    
    def validation_step(self, batch, batch_index):
        x, y = batch
        student_logits, teacher_logits = self(x)

        student_loss = F.cross_entropy(student_logits, y)
        teacher_loss = F.cross_entropy(teacher_logits, y)

        student_preds = torch.argmax(student_logits, dim=1)
        student_acc = accuracy(student_preds, y)
        teacher_preds = torch.argmax(teacher_logits, dim=1)
        teacher_acc = accuracy(teacher_preds, y)

        self.log("student_val_loss", student_loss, prog_bar=True)
        self.log("student_val_acc", student_acc, prog_bar=True)
        self.log("teacher_val_acc", teacher_acc, prog_bar=True)  # Make sure the teacher is properly trained and loaded

        return student_loss

    def configure_optimizers(self):
        return optim.Adam(self.student.parameters(), lr=0.01)
    
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