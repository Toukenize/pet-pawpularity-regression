import pandas as pd
from pytorch_lightning import Trainer, callbacks, seed_everything

from pawpularity.config.constants import (DROPOUT, LIN_HIDDEN, LR, META_COLS,
                                          MODEL_NAME, N_SPLITS, NUM_EPOCHS,
                                          OUT_DIR, TRAIN_BATCH_SIZE, TRAIN_CSV,
                                          TRAIN_IMG_DIR)
from pawpularity.model.callback import get_checkpoint_callback
from pawpularity.model.data import (bin_paw_train_target, get_dataloader,
                                    get_xth_split)
from pawpularity.model.model import PawImgModel


def train_model():

    df = pd.read_csv(TRAIN_CSV)
    df = bin_paw_train_target(df)
    seed_everything(2021, workers=True)

    for i in range(N_SPLITS):
        train_idx, val_idx = get_xth_split(
            df.index, df['bin'], split_num=i, n_splits=N_SPLITS)

        train_loader = get_dataloader(
            df=df.iloc[train_idx],
            img_folder=TRAIN_IMG_DIR,
            batch_size=TRAIN_BATCH_SIZE,
            is_train=True,
            shuffle=True,
            meta_cols=META_COLS,
            label_col='bin'
        )

        val_loader = get_dataloader(
            df=df.iloc[val_idx],
            img_folder=TRAIN_IMG_DIR,
            batch_size=TRAIN_BATCH_SIZE,
            is_train=False,
            shuffle=False,
            meta_cols=META_COLS,
            label_col='bin'
        )

        model = PawImgModel(
            model_name=MODEL_NAME,
            dropout=DROPOUT,
            meta_col_dim=len(META_COLS),
            pretrained=True,
            optim_configs=dict(lr=LR)
        )

        checkpoint_callback = get_checkpoint_callback(
            out_dir=OUT_DIR,
            file_prefix=f'fold_{i}',
            monitor='val_bce_loss',
            save_top_k=1,
            mode='min',
            save_weights_only=True
        )

        trainer = Trainer(
            gpus=1, max_epochs=NUM_EPOCHS,
            callbacks=[checkpoint_callback]
        )

        trainer.fit(
            model, train_dataloader=train_loader,
            val_dataloaders=val_loader
        )


if __name__ == "__main__":
    train_model()
