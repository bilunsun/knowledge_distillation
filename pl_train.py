import argparse
import os
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy


from data_utils import get_mnist_train_test
from neural_network_utils import StudentNet, TeacherNet


MODELS_PATH = ".models/"


class Model(pl.LightningModule):
    def __init__(self, net):
        super().__init__()

        self.net = net

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_index):
        x, y = batch
        logits = self.net(x)
        loss = F.cross_entropy(logits, y)

        return loss
    
    def validation_step(self, batch, batch_index):
        x, y = batch
        logits = self.net(x)
        loss = F.cross_entropy(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return optim.Adam(self.net.parameters(), lr=1e-3)


def main(who):
    if not os.path.exists(MODELS_PATH):
        os.makedirs(MODELS_PATH)

    train_loader, test_loader = get_mnist_train_test()

    net = StudentNet() if who == "student" else TeacherNet()

    model = Model(net)
    trainer = pl.Trainer(max_epochs=3, gpus=1)
    trainer.fit(model, train_loader, test_loader)

    torch.save(model.net.state_dict(), MODELS_PATH + f"{who}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--who",
                        help="Train the teacher or student network",
                        choices=("student", "teacher"),
                        required=True,
                        type=str)

    args = parser.parse_args()

    main(args.who)