from pytorch_lightning.callbacks import ModelCheckpoint


def get_checkpoint_callback(out_dir, file_prefix, **params):

    checkpoint_callback = ModelCheckpoint(
        dirpath=out_dir,
        filename=f"{file_prefix}_" +
        "{epoch:02d}_{val_bce_loss:.4f}_" +
        "{val_rmse_loss:.4f}",
        save_top_k=1,
    )

    return checkpoint_callback
