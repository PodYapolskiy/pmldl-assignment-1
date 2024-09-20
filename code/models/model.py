import torch
from torch import nn, optim, Tensor
import torch.nn.functional as F
import lightning as L

device = "cuda" if torch.cuda.is_available() else "cpu"


from torchmetrics import Accuracy


class MNISTClassifier(L.LightningModule):
    def __init__(self):
        super().__init__()

        # 28 x 28 x 1
        self.lin = nn.Sequential(
            nn.Linear(28 * 28, 128, device=device),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64, device=device),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 10, device=device),
            nn.Softmax(dim=1),
        )
        self.accuracy = Accuracy(task="binary")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch

        x = x.view(-1, 28 * 28)
        x = self.lin(x)

        loss = F.cross_entropy(x, y)
        acc = self.accuracy(x, y)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True)

        return loss

    # def training_epoch_end(self, outputs) -> None:
    #     loss = sum(output['loss'] for output in outputs) / len(outputs)
    #     print(loss)

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch

        x = x.view(-1, 28 * 28)
        x = self.lin(x)

        loss = F.cross_entropy(x, y)
        acc = self.accuracy(x, y)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch

        x = x.view(-1, 28 * 28)
        x = self.lin(x)

        test_loss = F.cross_entropy(x, y)
        self.log("val_loss", test_loss)

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        pred = self.lin(x)
        return pred

    def forward(self, x: Tensor) -> Tensor:
        """Used for model(x) inference"""
        with torch.no_grad():
            x = x.view(-1, 28 * 28)
            x = self.lin(x)
            return x
