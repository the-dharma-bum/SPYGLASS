from .. import SpyGlassDataModule

def init_datamodule(cfg):
    return SpyGlassDataModule(cfg.video_root,
                              cfg.channels,
                              cfg.x_size,
                              cfg.y_size,
                              cfg.sampling,
                              cfg.medical_data_csv_path, 
                              cfg.train_batch_size,
                              cfg.val_batch_size,
                              cfg.num_workers,
                              cfg.criteria)