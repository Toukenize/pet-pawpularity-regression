import logging
from typing import Dict, Optional

import timm
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.optim import AdamW

from .scheduler import (get_constant_schedule_with_warmup,
                        get_cosine_schedule_with_warmup,
                        get_step_schedule_with_warmup)


class PawImgModel(LightningModule):

    def __init__(
            self,
            model_name: str,
            dropout: float,
            optim_configs: Dict,
            # Validation done externally
            scheduler_configs: Optional[Dict] = None,
            meta_col_dim: int = 0,
            ** params):

        super().__init__()
        self.model = timm.create_model(model_name, **params)
        self.dropout1 = nn.Dropout(dropout)

        # Append a new linear layer on top of swin model (including imagenet head)
        self.meta_col_dim = meta_col_dim
        lin_w_meta = 1_000 + meta_col_dim
        self.out = nn.Linear(lin_w_meta, 1)

        # Criterion
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()

        # Optimizers
        self.optim_config = optim_configs
        self.scheduler_configs = scheduler_configs
        self.save_hyperparameters()

    def forward(self, image, meta):
        img_out = self.model(image)
        img_out = self.dropout1(img_out)

        if self.meta_col_dim > 0:
            out = torch.cat([img_out, meta], dim=-1)
        else:
            out = img_out

        out = self.out(out)

        return out

    def training_step(self, batch, _batch_idx):

        target = batch.pop('label')
        logits = self.forward(**batch)
        bce_loss = self.bce_loss(logits, target)

        rmse_loss = torch.sqrt(
            self.mse_loss(logits.sigmoid() * 100, target * 100)
        )

        self.log('bce_loss', bce_loss, prog_bar=True)
        self.log('rmse_loss', rmse_loss, prog_bar=True)

        return bce_loss

    def validation_step(self, batch, _batch_idx):

        target = batch.pop('label')
        logits = self.forward(**batch)
        bce_loss = self.bce_loss(logits, target)
        rmse_loss = torch.sqrt(
            self.mse_loss(logits.sigmoid() * 100, target * 100)
        )
        self.log('val_bce_loss', bce_loss, prog_bar=True)
        self.log('val_rmse_loss', rmse_loss, prog_bar=True)

        return bce_loss

    def configure_optimizers(self):

        opt = AdamW(self.parameters(), lr=self.optim_config["lr"])

        # Get warmup scheduler
        if self.scheduler_configs is not None:
            sch_type = self.scheduler_configs.pop('scheduler')

            if sch_type == 'cosine_warmup':
                sch_func = get_cosine_schedule_with_warmup

            elif sch_type == 'constant_warmup':
                sch_func = get_constant_schedule_with_warmup

            elif sch_type == 'step_warmup':
                sch_func = get_step_schedule_with_warmup

            lr_warmup = sch_func(
                opt, **self.scheduler_configs
            )

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": lr_warmup,
                "interval": "epoch",
                "frequency": 1,
                "name": f"LR Scheduler ({sch_type})"
            }
        }