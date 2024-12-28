import argparse
import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import chess
import torch
from torch.utils.data import ConcatDataset, DataLoader
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, Callback

from dataset import ChessDataset, ValidationDataset
from nn import load_model

BATCH_SIZE = 128


class ChessLightningModule(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        model_conf = config["model_conf"]
        if isinstance(model_conf, TransferConf):
            self.teacher = load_model(
                n_res_blocks=model_conf.teacher,
                checkpoint=config["last_ckpt"],
                inference=True,
                compile=config["compile_model"],
            )
            self.model = load_model(
                n_res_blocks=model_conf.student,
                inference=False,
                compile=config["compile_model"],
            )

        else:
            assert isinstance(model_conf, int)
            self.model = load_model(
                n_res_blocks=model_conf,
                checkpoint=config["last_ckpt"],
                inference=False,
                compile=config["compile_model"],
            )

    def compute_loss1(self, dist_pred, dist_gt):
        batch_size = dist_gt.shape[0]
        return -(dist_pred * dist_gt).sum() / batch_size

    def compute_loss2(self, value_pred, value_gt):
        batch_size = value_gt.shape[0]
        return F.mse_loss(value_pred, value_gt, reduction="sum") / batch_size

    def training_step(self, batch, batch_idx):
        boards, dist, outcome = batch

        if not isinstance(self.config["model_conf"], TransferConf):
            dist_pred, value_pred = self.model(boards)
            loss1 = self.compute_loss1(dist_pred, dist)
            loss2 = self.compute_loss2(value_pred, outcome)

        else:
            batch_size_supervised = self.config["batch_size"] // 2

            boards_supervised = boards[:batch_size_supervised]
            dist_supervised = dist[:batch_size_supervised]
            outcome_supervised = outcome[:batch_size_supervised]

            boards_unsupervised = boards[batch_size_supervised:]
            dist_unsupervised, outcome_unsupervised = self.teacher(boards_unsupervised)
            dist_unsupervised = torch.exp(dist_unsupervised)

            dist_student = torch.cat((dist_supervised, dist_unsupervised), dim=0)
            outcome_student = torch.cat(
                (outcome_supervised, outcome_unsupervised), dim=0
            )

            dist_pred, value_pred = self.model(boards)
            loss1 = self.compute_loss1(dist_pred, dist_student)
            loss2 = self.compute_loss2(value_pred, outcome_student)

        self.log_dict(
            {
                "loss1": loss1,
                "loss2": loss2,
                "loss": loss1 + self.config["loss_weight"] * loss2,
            }
        )
        return loss1 + self.config["loss_weight"] * loss2

    def validation_step(self, batch, batch_idx):
        boards, dist, outcome = batch
        dist_pred, value_pred = self.model(boards)
        loss1 = self.compute_loss1(dist_pred, dist)
        loss2 = self.compute_loss2(value_pred, outcome)
        loss = loss1 + self.config["loss_weight"] * loss2
        self.log_dict({"val_loss1": loss1, "val_loss2": loss2, "val_loss": loss})
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.config["lr"], weight_decay=1e-4
        )
        # return optimizer

        from torch.optim.lr_scheduler import OneCycleLR

        scheduler = OneCycleLR(
            optimizer=optimizer,
            max_lr=self.config["lr"],
            total_steps=self.config["steps_per_epoch"] * self.config["epochs"],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


class ModelCheckpointAtEpochEnd(Callback):
    def __init__(self, start, end, interval, model_compiled=True):
        super().__init__()
        self.start = start
        self.end = end
        self.interval = interval
        self.model_compiled = model_compiled

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        callback_metrics = trainer.callback_metrics

        if epoch < self.start or epoch >= self.end or (epoch + 1) % self.interval != 0:
            return

        path = os.path.join(
            trainer.log_dir,
            f"epoch:{epoch}-val_loss:{callback_metrics['val_loss']:0.3f}.ckpt",
        )
        print("saving checkpoint: ", path)

        m = pl_module.model
        if self.model_compiled:
            m = m._orig_mod

        torch.save(m.state_dict(), path)


@dataclass
class TransferConf:
    student: int
    teacher: int

    @classmethod
    def parse(cls, conf: Optional[str]) -> Optional["TransferConf"]:
        if conf is None:
            return None

        ts = conf.split(":")
        if len(ts) != 2:
            return None

        return cls(int(ts[1]), int(ts[0]))


def main():
    torch.set_float32_matmul_precision("medium")
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--trace-file", nargs="*", action="extend")
    parser.add_argument("-n", "--epochs", type=int, default=10)
    parser.add_argument("-c", "--last-ckpt", type=str)
    parser.add_argument("-l", "--lr", type=float, default=1e-4)
    parser.add_argument("-w", "--loss-weight", type=float, default=0.002)
    parser.add_argument("--save-every-k", type=int, default=10)
    parser.add_argument("--save-start", type=int, default=10)
    parser.add_argument("--save-end", type=int, default=100)
    parser.add_argument("--model-conf", type=str, default=None)
    parser.add_argument("--val-data", type=str, default="py/validation/sample.csv")
    args = parser.parse_args()

    compile_model = True

    if not args.trace_file:
        print("No trace file specified.")
        return

    logger = TensorBoardLogger("tb_logs", name="chess")
    lightning_checkpoints = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpointAtEpochEnd(
            args.save_start,
            args.save_end,
            args.save_every_k,
            model_compiled=compile_model,
        ),
    ]

    dss = ConcatDataset([ChessDataset(f) for f in args.trace_file])

    train_loader = DataLoader(
        dss, num_workers=4, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )

    val_loader = DataLoader(
        ValidationDataset(args.val_data),
        num_workers=4,
        batch_size=8,
        shuffle=False,
        drop_last=True,
    )

    config = dict(
        batch_size=BATCH_SIZE,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        lr=args.lr,
        trace_files=args.trace_file,
        last_ckpt=args.last_ckpt,
        save_start=args.save_start,
        save_end=args.save_end,
        save_every_k=args.save_every_k,
        loss_weight=args.loss_weight,
        compile_model=compile_model,
        model_conf=TransferConf.parse(args.model_conf) or int(args.model_conf),
    )

    module = ChessLightningModule(config)
    trainer = L.Trainer(
        enable_checkpointing=False,
        logger=logger,
        callbacks=lightning_checkpoints,
        max_epochs=config["epochs"],
        log_every_n_steps=5,
        precision="16-mixed",
    )
    trainer.fit(
        model=module, train_dataloaders=train_loader, val_dataloaders=val_loader
    )

    # m = module.model._orig_mod if compile_model else module.model
    # torch.save(m.state_dict(), os.path.join(trainer.log_dir, "last.ckpt"))


if __name__ == "__main__":
    main()
