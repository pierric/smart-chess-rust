import argparse
import json

import numpy as np
import chess
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor

import libencoder
from nn import load_model


def _get_outcome(res):
    if res is None:
        return 0.

    if res["winner"] is None:
        return 0.

    if res["winner"] == "White":
        return 1.

    return -1.


class ChessDataset(Dataset):
    def __init__(self, trace_file):
        with open(trace_file, "r") as f:
            trace = json.load(f)

        self.outcome = _get_outcome(trace["outcome"])
        steps = [
            (chess.Move.from_uci(step[0]), [c[0] for c in step[2]])
            for step in trace["steps"]
        ]
        self.steps = libencoder.encode(steps)


    def __len__(self):
        return len(self.steps)

    def __getitem__(self, idx):
        boards, meta, dist = self.steps[idx][:3]
        inp = np.concatenate((boards, meta), axis=-1).astype(np.float32)
        inp = inp.transpose((2, 0, 1))
        return torch.from_numpy(inp), torch.from_numpy(dist), torch.tensor([self.outcome], dtype=torch.float32)


class ChessLightningModule(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.model = load_model(checkpoint=config["last_ckpt"])
        self.config = config

    def training_step(self, batch, batch_idx):
        boards, dist, outcome = batch
        dist_pred, value_pred = self.model(boards)
        loss1 = F.kl_div(dist_pred, dist, reduction="batchmean")
        loss2 = F.mse_loss(value_pred, outcome)
        self.log_dict({
            "loss1": loss1,
            "loss2": loss2,
        })
        return loss1 + loss2

    def configure_optimizers(self):
        from torch.optim.lr_scheduler import OneCycleLR
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config["lr"])
        scheduler = OneCycleLR(
            optimizer=optimizer,
            max_lr=self.config["lr"],
            total_steps=self.config["steps_per_epoch"] * self.config["epochs"]
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--trace-file", action="append")
    parser.add_argument("-n", "--epochs", type=int, default=4)
    parser.add_argument("-c", "--last-ckpt", type=str)
    args = parser.parse_args()

    if not args.trace_file:
        print("No trace file specified.")
        return

    logger = TensorBoardLogger("tb_logs", name="chess")
    lr_monitor = LearningRateMonitor(logging_interval='step')

    dss = ConcatDataset([ChessDataset(f) for f in args.trace_file])
    train_loader = DataLoader(dss, num_workers=4, batch_size=2, shuffle=True)

    config = dict(
        epochs = args.epochs,
        steps_per_epoch = len(train_loader),
        lr = 1e-3,
        trace_files = args.trace_file,
        last_ckpt = args.last_ckpt,
    )

    module = ChessLightningModule(config)
    trainer = L.Trainer(logger=logger, callbacks=[lr_monitor], max_epochs=config["epochs"], log_every_n_steps=5)
    trainer.fit(model=module, train_dataloaders=train_loader)
    # trainer.save_checkpoint("last.ckpt")


if __name__ == "__main__":
    main()
