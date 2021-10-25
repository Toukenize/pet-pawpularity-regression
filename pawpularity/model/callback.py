from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)


def get_callbacks(
        out_dir=None, file_prefix='',
        logging_interval='step',
        monitor='val_rmse_loss',
        mode='min',
        **early_stop_params
):

    callbacks = []

    if out_dir is not None:
        callbacks.append(
            ModelCheckpoint(
                dirpath=out_dir,
                filename=f"{file_prefix}_" +
                "{epoch:02d}_{val_bce_loss:.4f}_" +
                "{val_rmse_loss:.4f}",
                save_top_k=1,
                save_weights_only=True,
                monitor=monitor,
                mode=mode
            )
        )

    if logging_interval is not None:

        callbacks.append(
            LearningRateMonitor(logging_interval=logging_interval)
        )

    if monitor is not None:

        callbacks.append(
            EarlyStopping(monitor=monitor, mode=mode, **early_stop_params)
        )

    return callbacks
