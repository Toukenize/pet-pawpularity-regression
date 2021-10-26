import pandas as pd
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger

import wandb
from pawpularity.config.constants import (DROPOUT, LR, META_COLS, MODEL_NAME,
                                          N_SPLITS, NUM_EPOCHS,
                                          NUM_VAL_PER_EPOCH, OUT_DIR, RUN_NAME,
                                          TRAIN_BATCH_SIZE, TRAIN_CSV,
                                          TRAIN_IMG_DIR, WANDB_ENTITY,
                                          WANDB_PROJECT)
from pawpularity.model.callback import get_callbacks
from pawpularity.model.data import (bin_paw_train_target, get_dataloader,
                                    get_xth_split)
from pawpularity.model.model import PawImgModel


def train_model():

    df = pd.read_csv(TRAIN_CSV)
    df = bin_paw_train_target(df)
    seed_everything(2021, workers=True)

    for i in range(N_SPLITS):

        logger = WandbLogger(
            name=f'Fold_{i}|{RUN_NAME}',
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            group=RUN_NAME)

        train_idx, val_idx = get_xth_split(
            df.index, df['bin'], split_num=i, n_splits=N_SPLITS)

        train_loader = get_dataloader(
            df=df.iloc[train_idx],
            img_folder=TRAIN_IMG_DIR,
            batch_size=TRAIN_BATCH_SIZE,
            is_train=True,
            shuffle=True,
            meta_cols=META_COLS,
            label_col='Pawpularity'
        )

        val_loader = get_dataloader(
            df=df.iloc[val_idx],
            img_folder=TRAIN_IMG_DIR,
            batch_size=TRAIN_BATCH_SIZE,
            is_train=False,
            shuffle=False,
            meta_cols=META_COLS,
            label_col='Pawpularity'
        )

        model = PawImgModel(
            model_name=MODEL_NAME,
            dropout=DROPOUT,
            meta_col_dim=len(META_COLS),
            pretrained=True,
            optim_configs=dict(lr=LR)
        )

        callbacks = get_callbacks(
            out_dir=OUT_DIR,
            file_prefix=f'fold_{i}',
            monitor='val_rmse_loss',
            mode='min',
            patience=6
        )

        trainer = Trainer(
            gpus=1,
            val_check_interval=len(train_loader) // NUM_VAL_PER_EPOCH,
            max_epochs=NUM_EPOCHS,
            callbacks=callbacks,
            logger=logger
        )

        trainer.fit(
            model, train_dataloaders=train_loader,
            val_dataloaders=val_loader
        )

        wandb.finish()


if __name__ == "__main__":
    train_model()
