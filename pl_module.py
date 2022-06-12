import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import torchmetrics
from torchmetrics.functional import accuracy
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import create_model

class ClassifierNet(pl.LightningModule):
    def __init__(self, lr=0.01, model_name="resnet34", num_classes=3):
        super().__init__()
        self.save_hyperparameters()
        self.model = create_model(model_name=model_name)
        self.confmatrix = torchmetrics.ConfusionMatrix(num_classes=num_classes)
        self.print_matrix_every_epochs = 10

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    ## Invididual train/validation/test steps
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        self.log(f"train_acc", acc, prog_bar=True)
        self.log(f"train_loss", loss, prog_bar=True)

        return {
            'loss': loss,
            'train_loss': loss,
            'train_acc': acc,
            'preds': preds,
            'y': y,
        }

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss, acc, preds = self._shared_val_step(batch, batch_idx)
        return self._shared_log_step(
            'val', loss, acc, preds, y
        )

    def test_step(self, batch, batch_idx):
        x, y = batch
        loss, acc, preds = self._shared_val_step(batch, batch_idx)
        return self._shared_log_step(
            'test', loss, acc, preds, y
        )

    def _shared_val_step(self, batch, batch_idx):
        """Shared val/test step to run the model and get predictions"""
        x, y = batch
        y_hat = self(x)
        loss = F.nll_loss(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = accuracy(preds, y)
        return loss, acc, preds

    def _shared_log_step(self, stage, loss, acc, preds, y):
        """Log and return the metrics for a val/test step"""
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_epoch=True)

        return {
            f'{stage}_acc': acc,
            f'{stage}_loss': loss,
            'preds': preds,
            'y': y,
        }

    # ON EPOCH END METHODS
    def training_epoch_end(self, train_step_outputs):
        self.epoch_end_log("train", train_step_outputs)

    def validation_epoch_end(self, validation_step_outputs):
        self.epoch_end_log("val", validation_step_outputs)

    def test_epoch_end(self, test_step_outputs):
        self.epoch_end_log("test", test_step_outputs)

    def epoch_end_log(self, stage, output):
        """
        Called at the end of 'train'|'val'|'test' epoch.
        Print the confusion matrix after 'print_matrix_every_epochs'
        epochs. Or every time it is called for a 'test' epoch.

        :param str stage: ['train', 'val', 'test']
        :param list of dicts output: list of dicts. Automatic summary of
            the outputs returned by the individual steps
        :return: None
        """
        preds = torch.cat([val['preds'] for val in output], dim=0)
        y = torch.cat([val['y'] for val in output], dim=0)
        acc = accuracy(preds, y)

        if stage == 'test' or (
                self.current_epoch > 1 and
                self.current_epoch % self.print_matrix_every_epochs == 0
        ):
            cmatrix = self.confmatrix(preds, y)
            print(f'{stage} Epoch {self.current_epoch + 1}: Acc: {acc:.4f}')
            print(f"{stage} Confusion Matrix = {cmatrix}")
            if stage == 'train':
                print("=" * 50)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Prediction step"""
        x, y = batch
        y_hat = self(x)
        preds = torch.argmax(y_hat, dim=1)
        return preds

    def configure_optimizers(self):
        """
        See:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
        """
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler_dict = {
            "scheduler": ReduceLROnPlateau(
                optimizer,
                factor=0.25,
                threshold=0.001,
                threshold_mode='rel',
                cooldown=3,
            ),
            "interval": "epoch",
            "monitor": "val_acc",
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_dict,
        }
