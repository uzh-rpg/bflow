import os
from pathlib import Path

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.loggers import CSVLogger
import torch

from modules.data_loading import DataModule
from modules.raft_spline import RAFTSplineModule


@hydra.main(config_path='config', config_name='val', version_base='1.3')
def main(config: DictConfig):
    print('------ Configuration ------\n')
    print(OmegaConf.to_yaml(config))
    print('---------------------------\n')
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)

    # ------------
    # GPU Options
    # ------------
    gpus = config.hardware.gpus
    assert isinstance(gpus, int), 'no more than 1 gpu supported'
    gpus = [gpus]

    batch_size: int = config.batch_size
    assert batch_size > 0

    # ------------
    # Data
    # ------------
    data_module = DataModule(config, batch_size_train=batch_size, batch_size_val=batch_size)

    num_bins_context = data_module.get_nbins_context()
    num_bins_corr = data_module.get_nbins_correlation()
    print(f'num_bins:\n\tcontext: {num_bins_context}\n\tcorrelation: {num_bins_corr}')

    # ---------------------
    # Logging and Checkpoints
    # ---------------------
    logger = CSVLogger(save_dir='./validation_logs')
    ckpt_path = Path(config.checkpoint)

    # ------------
    # Model
    # ------------

    module = RAFTSplineModule.load_from_checkpoint(str(ckpt_path), **{'config': config})

    # ---------------------
    # Callbacks and Misc
    # ---------------------
    callbacks = [ModelSummary(max_depth=2)]

    trainer = pl.Trainer(
        accelerator='gpu',
        callbacks=callbacks,
        default_root_dir=None,
        devices=gpus,
        logger=logger,
        log_every_n_steps=100,
        precision=32,
    )

    with torch.inference_mode():
        trainer.validate(model=module, datamodule=data_module, ckpt_path=str(ckpt_path))


if __name__ == '__main__':
    main()
