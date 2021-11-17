import logging

import pandas as pd
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger

import wandb
from pawpularity.config.constants import (DROPOUT, EARLY_STOP_PATIENCE, LR,
                                          META_COLS, MODEL_NAME, N_SPLITS,
                                          NUM_CYCLES, NUM_EPOCHS,
                                          NUM_VAL_PER_EPOCH, NUM_WARMUP_EPOCHS,
                                          OUT_DIR, RUN_NAME, SCHEDULER,
                                          STEP_FACTOR, STEP_SIZE, TRAIN_CSV,
                                          WANDB_ENTITY, WANDB_PROJECT)
from pawpularity.model.pytorch_lightning.callback import get_callbacks
from pawpularity.model.pytorch_lightning.model import PawImgModel
from pawpularity.model.pytorch_lightning.scheduler import ScheduleValidator
from pawpularity.processing import bin_paw_train_target, get_xth_split

logging.basicConfig(level=logging.INFO)


# TODO: Add arguments to control script behaviour
# e.g. sampling, number of folds

def train_model():

    df = pd.read_csv(TRAIN_CSV)

    # df = df.sample(frac=0.1, random_state=2021).reset_index(drop=True)

    logging.info(f'DF Shape : {df.shape}')

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

        if SCHEDULER is not None:
            sch_config = (
                ScheduleValidator(
                    scheduler=SCHEDULER,
                    num_warmup_steps=NUM_WARMUP_EPOCHS,
                    num_training_steps=NUM_EPOCHS - NUM_WARMUP_EPOCHS,
                    num_cycles=NUM_CYCLES,
                    step_size=STEP_SIZE,
                    step_factor=STEP_FACTOR
                )
                .dict(exclude_none=True)
            )
        else:
            sch_config = None

        model = PawImgModel(
            model_name=MODEL_NAME,
            dropout=DROPOUT,
            meta_col_dim=len(META_COLS),
            pretrained=True,
            optim_configs=dict(lr=LR),
            scheduler_configs=sch_config,
            df=df,
            train_idx=train_idx,
            val_idx=val_idx
        )

        model.prepare_data()

        callbacks = get_callbacks(
            out_dir=OUT_DIR,
            file_prefix=f'fold_{i}',
            logging_interval='epoch',
            monitor='val_rmse_loss',
            mode='min',
            patience=EARLY_STOP_PATIENCE
        )

        trainer = Trainer(
            gpus=1,
            auto_scale_batch_size="power",
            val_check_interval=1/NUM_VAL_PER_EPOCH,
            max_epochs=NUM_EPOCHS,
            callbacks=callbacks,
            logger=logger,
            enable_model_summary=False
        )
        trainer.tune(model)
        trainer.fit(model)

        # Log hyperparams after fitting model
        # So that batch size found automatically can be captured
        logger.log_hyperparams(model.hparams)

        wandb.finish()


if __name__ == "__main__":
    train_model()
