import argparse
import json
import os

import numpy as np
import chess
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, Callback

import libencoder
from nn import load_model

BATCH_SIZE = 128


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
            (chess.Move.from_uci(step[0]), [c[1] for c in step[2]])
            for step in trace["steps"]
        ]
        self.steps = libencoder.encode_steps(steps)


    def __len__(self):
        return len(self.steps)

    def __getitem__(self, idx):
        boards, meta, dist = self.steps[idx][:3]
        inp = np.concatenate((boards, meta), axis=-1).astype(np.float32)
        inp = inp.transpose((2, 0, 1))
        turn = meta[0, 0, 0]
        assert idx % 2 == 1 - turn
        outcome = self.outcome * (1 if turn == 1 else -1)
        return torch.from_numpy(inp), torch.from_numpy(dist), torch.tensor([outcome], dtype=torch.float32)


class ChessLightningModule(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.model = load_model(checkpoint=config["last_ckpt"])
        self.config = config

    def training_step(self, batch, batch_idx):
        boards, dist, outcome = batch
        dist_pred, value_pred = self.model(boards)

        loss1 = - (dist_pred * dist).sum() / self.config["batch_size"]

        loss2 = F.mse_loss(value_pred, outcome, reduction="sum") / self.config["batch_size"]
        self.log_dict({
            "loss1": loss1,
            "loss2": loss2,
        })
        return loss1 + 0.002 * loss2

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config["lr"], weight_decay=1e-4)
        # return optimizer

        from torch.optim.lr_scheduler import OneCycleLR
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

class ModelCheckpointAtEpochEnd(Callback):
    def __init__(self, interval):
        super().__init__()
        self.interval = interval

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch

        if (epoch + 1) % self.interval != 0:
            return

        path = os.path.join(trainer.log_dir, f"epoch-{epoch}.ckpt")
        print("saving checkpoint: ", path)
        torch.save(pl_module.model._orig_mod.state_dict(), path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--trace-file", nargs="*", action="extend")
    parser.add_argument("-n", "--epochs", type=int, default=100)
    parser.add_argument("-c", "--last-ckpt", type=str)
    parser.add_argument("-l", "--lr", type=float, default=1e-4)
    parser.add_argument("--save-every-k", type=int, default=10)
    args = parser.parse_args()

    if not args.trace_file:
        print("No trace file specified.")
        return

    logger = TensorBoardLogger("tb_logs", name="chess")
    lightning_checkpoints = [
        LearningRateMonitor(logging_interval='step'),
        ModelCheckpointAtEpochEnd(args.save_every_k),
    ]

    dss = ConcatDataset([ChessDataset(f) for f in args.trace_file])
    train_loader = DataLoader(dss, num_workers=4, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    config = dict(
        batch_size = BATCH_SIZE,
        epochs = args.epochs,
        steps_per_epoch = len(train_loader),
        lr = args.lr,
        trace_files = args.trace_file,
        last_ckpt = args.last_ckpt,
        save_every_k = args.save_every_k,
    )

    module = ChessLightningModule(config)
    trainer = L.Trainer(
        enable_checkpointing=False,
        logger=logger,
        callbacks=lightning_checkpoints,
        max_epochs=config["epochs"],
        log_every_n_steps=50,
    )
    trainer.fit(model=module, train_dataloaders=train_loader)

    torch.save(module.model._orig_mod.state_dict(), os.path.join(trainer.log_dir, "last.ckpt"))


if __name__ == "__main__":
    main()
