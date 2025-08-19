import argparse
import json
from math import exp, log
import os
from dataclasses import dataclass
from typing import Optional
from multiprocessing import Pool
from functools import partial
from torch.optim import optimizer
from tqdm import tqdm

import numpy as np
import chess
import torch
from torch.utils.data import ConcatDataset, DataLoader, random_split
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, Callback

from dataset import ChessDataset, ValidationDataset
from module import load_model


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
                compile=False,
            )
            self.model = load_model(
                n_res_blocks=model_conf.student,
                inference=False,
                compile=False,
            )

        else:
            assert isinstance(model_conf, int)
            self.model = load_model(
                n_res_blocks=model_conf,
                checkpoint=config["last_ckpt"],
                inference=False,
                compile=False,
                omit=config.get("omit"),
            )
            self.teacher = None

        freeze_prefix = []
        if config["freeze_backbone"]:
            freeze_prefix = ["model.conv_block.", "model.res_blocks."]

        if config["freeze"]:
            freeze_prefix.extend(config["freeze"])

        if freeze_prefix:
            trainable_weights = []

            for name, param in self.named_parameters():
                if not any(name.startswith(p) for p in freeze_prefix):
                    trainable_weights.append(name)
                    continue
                param.requires_grad = False
            print("Trainnable weights: ", trainable_weights)

        if config["compile_model"]:
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            self.model.compile(mode="reduce-overhead", fullgraph=True)
            # skip compiling the teacher, as we don't use do transfer-learning yet.

    def compute_loss1(self, log_dist_pred, dist_gt):
        batch_size = dist_gt.shape[0]
        return -(log_dist_pred * dist_gt).sum() / batch_size

    def compute_loss2(self, value_pred, value_gt):
        batch_size = value_gt.shape[0]
        return F.mse_loss(value_pred, value_gt, reduction="sum") / batch_size

    def training_step(self, batch, batch_idx):
        boards, dist, outcome = batch

        if not isinstance(self.config["model_conf"], TransferConf):
            log_dist_pred, value_pred = self.model(boards)
            loss1 = self.compute_loss1(log_dist_pred, dist)
            loss2 = self.compute_loss2(value_pred, outcome)

        else:
            batch_size_supervised = self.config["train_batch_size"] // 2

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

            log_dist_pred, value_pred = self.model(boards)
            loss1 = self.compute_loss1(log_dist_pred, dist_student)
            loss2 = self.compute_loss2(value_pred, outcome_student)

        self.log_dict(
            {
                "loss1": loss1,
                "loss2": loss2,
                "loss": loss1 + self.config["loss_weight"] * loss2,
                "w2": torch.nn.utils.get_total_norm(self.parameters()),
            }
        )
        return loss1 + self.config["loss_weight"] * loss2

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        prefix = "val_syn" if dataloader_idx == 0 else "val_real"
        boards, dist, outcome = batch
        log_dist_pred, value_pred = self.model(boards)
        loss1 = self.compute_loss1(log_dist_pred, dist)
        loss2 = self.compute_loss2(value_pred, outcome)

        loss1, loss2 = self.all_gather([loss1, loss2])
        if self.trainer.is_global_zero:
            loss1 = loss1.mean(dim=0)
            loss2 = loss2.mean(dim=0)
            loss = loss1 + self.config["loss_weight"] * loss2
            self.log_dict(
                {
                    f"{prefix}_loss1": loss1,
                    f"{prefix}_loss2": loss2,
                    f"{prefix}_loss": loss,
                }
            )

        self.trainer.strategy.barrier()

    def configure_optimizers(self):
        optm = self.config.get("optimizer", "AdamW")

        if optm == "AdamW":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.config["lr"],
                weight_decay=self.config["weight_decay"],
            )

        elif optm == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.config["lr"],
                momentum=0.9,
                weight_decay=self.config["weight_decay"],
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optm}")

        if self.config["lr_scheduler"] == "constant":
            return optimizer

        from torch.optim.lr_scheduler import OneCycleLR, ExponentialLR

        sched = self.config["lr_scheduler"]
        if sched == "exponential":
            gamma = exp(log(0.2) / self.config["epochs"])
            print(f"using gamma: {gamma:.4f}")
            scheduler = ExponentialLR(optimizer=optimizer, gamma=gamma)
            interval = "epoch"

        elif sched == "onecycle":
            scheduler = OneCycleLR(
                optimizer=optimizer,
                max_lr=self.config["lr"],
                total_steps=self.config["steps_per_epoch"] * self.config["epochs"],
            )
            interval = "step"

        else:
            raise RuntimeError("unknown scheduler: " + sched)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": interval,
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
        self.counting = 0

    def on_validation_end(self, trainer, pl_module):
        step = trainer.global_step
        callback_metrics = trainer.callback_metrics
        self.counting += 1

        if step < self.start or step >= self.end or self.counting < self.interval:
            return

        self.counting = 0

        val_loss1 = callback_metrics["val_syn_loss1/dataloader_idx_0"].item()
        val_loss2 = callback_metrics["val_syn_loss2/dataloader_idx_0"].item()

        path = os.path.join(
            trainer.log_dir or ".",
            f"step:{step}-{val_loss1:0.3f}-{val_loss2:0.3f}.ckpt",
        )
        print("saving checkpoint: ", path)

        m = pl_module.model
        ## necessary only if calling torch.compile instead of model.compile
        # if self.model_compiled:
        #    m = m._orig_mod

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
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--trace-file-for-train", nargs="+", action="extend")
    parser.add_argument("-v", "--trace-file-for-val", nargs="+", action="extend")
    parser.add_argument("-n", "--epochs", type=int, default=10)
    parser.add_argument("--max-epochs", type=int, default=0)
    parser.add_argument("-c", "--last-ckpt", type=str)
    parser.add_argument("-l", "--lr", type=float, default=1e-4)
    parser.add_argument("-w", "--loss-weight", type=float, default=0.002)
    parser.add_argument("--save-every-k", type=int, default=1)
    parser.add_argument("--save-start", type=int, default=10)
    parser.add_argument("--save-end", type=int, default=0)
    parser.add_argument("--model-conf", type=str, default=None)
    parser.add_argument("--real-play", type=str, default="py/validation/sample.csv")
    parser.add_argument("--train-batch-size", type=int, default=1024)
    parser.add_argument("--val-batch-size", type=int, default=128)
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        choices=["constant", "exponential", "onecycle"],
        default="constant",
    )
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--val-check-interval", type=int, default=0)
    parser.add_argument("--gradient-clip-val", type=float, default=0)
    parser.add_argument("--optimizer", type=str, default="AdamW")

    parser.add_argument("--omit", type=str, nargs="*")
    parser.add_argument("--freeze", type=str, nargs="*")
    args = parser.parse_args()

    compile_model = not args.no_compile

    if not args.trace_file_for_train:
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

    with Pool(12) as p:
        dss = list(
            tqdm(
                p.imap(partial(ChessDataset), args.trace_file_for_train),
                total=len(args.trace_file_for_train),
            )
        )
    train_split = ConcatDataset(dss)

    # train_split, val_syn_split = random_split(dss, [0.8, 0.2])

    train_loader = DataLoader(
        train_split,
        num_workers=4,
        batch_size=args.train_batch_size,
        shuffle=True,
        drop_last=True,
        persistent_workers=True,
    )

    with Pool(12) as p:
        dss = list(
            tqdm(
                p.imap(partial(ChessDataset), args.trace_file_for_val),
                total=len(args.trace_file_for_val),
            )
        )
    val_syn_split = ConcatDataset(dss)

    val_syn = DataLoader(
        val_syn_split,
        num_workers=4,
        batch_size=args.val_batch_size,
        shuffle=False,
        drop_last=True,
        persistent_workers=True,
    )

    val_real = DataLoader(
        ValidationDataset(args.real_play),
        num_workers=4,
        batch_size=args.val_batch_size,
        shuffle=False,
        drop_last=True,
        persistent_workers=True,
    )

    config = dict(
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        epochs=args.epochs,
        max_epochs=args.epochs if args.max_epochs == 0 else args.max_epochs,
        steps_per_epoch=len(train_loader),
        lr=args.lr,
        trace_file_for_train=args.trace_file_for_train,
        trace_file_for_val=args.trace_file_for_val,
        last_ckpt=args.last_ckpt,
        save_start=args.save_start,
        save_end=args.save_end,
        save_every_k=args.save_every_k,
        loss_weight=args.loss_weight,
        compile_model=compile_model,
        model_conf=(
            None
            if args.model_conf is None
            else TransferConf.parse(args.model_conf) or int(args.model_conf)
        ),
        freeze_backbone=args.freeze_backbone,
        lr_scheduler=args.lr_scheduler,
        weight_decay=args.weight_decay,
        gradient_clip_val=args.gradient_clip_val,
        optimizer=args.optimizer,
        omit=args.omit or None,
        freeze=args.freeze or None,
    )

    module = ChessLightningModule(config)

    extra_params = {}
    if args.val_check_interval > 0:
        extra_params["val_check_interval"] = args.val_check_interval

    if args.gradient_clip_val > 0:
        extra_params["gradient_clip_val"] = args.gradient_clip_val

    trainer = L.Trainer(
        enable_checkpointing=False,
        logger=logger,
        callbacks=lightning_checkpoints,
        max_epochs=config["max_epochs"],
        log_every_n_steps=10,
        # precision="bf16-mixed",
        # val_check_interval=20,
        **extra_params,
    )
    trainer.fit(
        model=module,
        train_dataloaders=train_loader,
        val_dataloaders=[val_syn, val_real],
    )

    # m = module.model._orig_mod if compile_model else module.model
    m = module.model
    torch.save(m.state_dict(), os.path.join(trainer.log_dir, "last.ckpt"))


if __name__ == "__main__":
    main()
