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
from pytorch_lightning.callbacks import LearningRateMonitor, ModelSummary, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
import torch
import wandb

from callbacks.logger import WandBImageLoggingCallback
from utils.general import get_ckpt_callback
from loggers.wandb_logger import WandbLogger
from modules.data_loading import DataModule
from modules.raft_spline import RAFTSplineModule


@hydra.main(config_path='config', config_name='train', version_base='1.3')
def main(cfg: DictConfig):
    print('------ Configuration ------\n')
    print(OmegaConf.to_yaml(cfg))
    print('---------------------------\n')
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    # ------------
    # Args
    # ------------
    gpu_devices = config['hardware']['gpus']
    gpus = gpu_devices if isinstance(gpu_devices, list) else [gpu_devices]
    num_gpus = len(gpus)

    batch_size: int = config['training']['batch_size']
    assert batch_size > 0
    per_gpu_batch_size = batch_size
    if num_gpus == 1:
        strategy = 'auto'
        ddp_active = False
    else:
        strategy = DDPStrategy(process_group_backend='nccl',
                               find_unused_parameters=False,
                               gradient_as_bucket_view=True)
        ddp_active = True
        per_gpu_batch_size = batch_size // num_gpus
        assert_info = 'Batch size ({}) must be divisible by number of gpus ({})'.format(batch_size, num_gpus)
        assert batch_size * num_gpus == per_gpu_batch_size, assert_info

    limit_train_batches = float(config['training']['limit_train_batches'])
    if limit_train_batches > 1.0:
        limit_train_batches = int(limit_train_batches)

    limit_val_batches = float(config['training']['limit_val_batches'])
    if limit_val_batches > 1.0:
        limit_val_batches = int(limit_val_batches)

    # ------------
    # Data
    # ------------
    data_module = DataModule(
        config,
        batch_size_train=per_gpu_batch_size,
        batch_size_val=per_gpu_batch_size)

    num_bins_context = data_module.get_nbins_context()
    num_bins_corr = data_module.get_nbins_correlation()
    print(f'num_bins:\n\tcontext: {num_bins_context}\n\tcorrelation: {num_bins_corr}')

    # ------------
    # Logging
    # ------------
    wandb_config = config['wandb']
    wandb_runpath = wandb_config['wandb_runpath']
    if wandb_runpath is None:
        wandb_id = wandb.util.generate_id()
        print(f'new run: generating id {wandb_id}')
    else:
        wandb_id = Path(wandb_runpath).name
        print(f'using provided id {wandb_id}')
    logger = WandbLogger(
        project=wandb_config['project_name'],
        group=wandb_config['group_name'],
        wandb_id=wandb_id,
        log_model=True,
        save_last_only_final=False,
        save_code=True,
        config_args=config)
    resume_path = None
    if wandb_config['artifact_name'] is not None:
        artifact_runpath = wandb_config['artifact_runpath']
        if artifact_runpath is None:
            artifact_runpath = wandb_runpath
        if artifact_runpath is None:
            print(
                'must specify wandb_runpath or artifact_runpath to restore a checkpoint/artifact. Cannot load artifact.')
        else:
            artifact_name = wandb_config['artifact_name']
            print(f'resuming checkpoint from runpath {artifact_runpath} and artifact name {artifact_name}')
            resume_path = logger.get_checkpoint(artifact_runpath, artifact_name)
            assert resume_path.exists()
            assert resume_path.suffix == '.ckpt', resume_path.suffix

    # ------------
    # Checkpoints
    # ------------
    checkpoint_callback = get_ckpt_callback(config=config)

    # ------------
    # Other Callbacks
    # ------------
    image_callback = WandBImageLoggingCallback(config['logging'])

    callback_list = None
    if config['debugging']['profiler'] is None:
        callback_list = [checkpoint_callback, image_callback]
        if config['training']['lr_scheduler']['use']:
            callback_list.append(LearningRateMonitor(logging_interval='step'))
    # ------------
    # Model
    # ------------

    if resume_path is not None and wandb_config['resume_only_weights']:
        print('Resuming only the weights instead of the full training state')
        net = RAFTSplineModule.load_from_checkpoint(str(resume_path), **{"config": config})
        resume_path = None
    else:
        net = RAFTSplineModule(config)

    # ------------
    # Training
    # ------------
    logger.watch(net, log='all', log_freq=int(config['logging']['log_every_n_steps']), log_graph=True)

    gradient_clip_val = config['training']['gradient_clip_val']
    if gradient_clip_val is not None and gradient_clip_val > 0:
        for param in net.parameters():
            param.register_hook(lambda grad: torch.clamp(grad, -gradient_clip_val, gradient_clip_val))

    # ---------------------
    # Callbacks and Misc
    # ---------------------
    callbacks = list()
    if config['training']['lr_scheduler']['use']:
        callbacks.append(LearningRateMonitor(logging_interval='step'))
    callbacks.append(ModelSummary(max_depth=2))

    trainer = pl.Trainer(
        accelerator='gpu',
        strategy=strategy,
        enable_checkpointing=True,
        sync_batchnorm=True if ddp_active else False,
        devices=gpus,
        logger=logger,
        precision=32,
        max_epochs=int(config['training']['max_epochs']),
        max_steps=int(config['training']['max_steps']),
        profiler=config['debugging']['profiler'],
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        log_every_n_steps=int(config['logging']['log_every_n_steps']),
        callbacks=callback_list)
    trainer.fit(model=net, ckpt_path=resume_path, datamodule=data_module)


if __name__ == '__main__':
    main()
