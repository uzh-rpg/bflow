import random
from typing import Dict, Union, Any, Optional, List

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import wandb

import torch

from callbacks.utils.visualization import (
    create_summary_img,
    get_grad_flow_figure,
    multi_plot_bezier_array,
    ev_repr_reduced_to_img_grayscale,
    img_torch_to_numpy)
from utils.general import is_cpu
from data.utils.keys import DataLoading, DataSetType

from loggers.wandb_logger import WandbLogger
from models.raft_spline.bezier import BezierCurves


class WandBImageLoggingCallback(Callback):
    FLOW_GT: str = 'flow_gt_img'
    FLOW_PRED: str = 'flow_pred_img'
    FLOW_VALID: str = 'flow_valid_img'
    EV_REPR_REDUCED: str = 'ev_repr_reduced'
    EV_REPR_REDUCED_M1: str = 'ev_repr_reduced_m1'
    VAL_BATCH_IDX: str = 'val_batch_idx'
    BEZIER_PARAMS: str = 'bezier_prediction'
    IMAGES: str = 'images'

    MAX_FLOW_ERROR = {
        int(DataSetType.DSEC): 2.0,
        int(DataSetType.MULTIFLOW2D): 3.0,
    }

    def __init__(self, logging_params: Dict[str, Any], deterministic: bool=True):
        super().__init__()
        log_every_n_train_steps = logging_params['log_every_n_steps']
        log_n_val_predictions = logging_params['log_n_val_predictions']
        self.log_only_numbers = logging_params['only_numbers']
        assert log_every_n_train_steps > 0
        assert log_n_val_predictions > 0
        self.log_every_n_train_steps = log_every_n_train_steps
        self.log_n_val_predictions = log_n_val_predictions
        self.deterministic = deterministic

        self._clear_val_data()
        self._training_started = False
        self._val_batch_indices = None

        self._dataset_type: Optional[DataSetType] = None

    def enable_immediate_validation(self):
        self._training_started = True

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if self.log_only_numbers:
            return
        if not self._training_started:
            self._training_started = True
        global_step = trainer.global_step
        do_log = True
        do_log &= global_step >= self.log_every_n_train_steps
        # BUG:  wandb bug? If we log metrics on the same step as logging images in this function
        #       then no metrics are logged at all, only the images. Unclear why this happens.
        #       So we introduce a small step margin to ensure that we don't log at the same time metrics and what we log here.
        delta_step = 2
        do_log &= (global_step - delta_step) % self.log_every_n_train_steps == 0
        if not do_log:
            return

        if self._dataset_type is None:
            self._dataset_type = batch[DataLoading.DATASET_TYPE][0].cpu().item()

        logger: WandbLogger = trainer.logger

        if isinstance(outputs['gt'], list):
            flow_gt = [x.detach().cpu() for x in outputs['gt']]
        else:
            flow_gt = outputs['gt'].detach().cpu()
        flow_pred = outputs['pred'].detach().cpu()
        flow_valid = outputs['gt_valid'].detach().cpu() if 'gt_valid' in outputs else None

        ev_repr_reduced = None
        if 'ev_repr_reduced' in outputs.keys():
            ev_repr_reduced = outputs['ev_repr_reduced'].detach().cpu()
        ev_repr_reduced_m1 = None
        if 'ev_repr_reduced_m1' in outputs.keys():
            ev_repr_reduced_m1 = outputs['ev_repr_reduced_m1'].detach().cpu()
        images = None
        if 'images' in outputs.keys():
            images = [x.detach().cpu() for x in outputs['images']]

        summary_img = create_summary_img(
            flow_pred,
            flow_gt[-1] if isinstance(flow_gt, list) else flow_gt,
            valid_mask=flow_valid,
            ev_repr_reduced=ev_repr_reduced,
            ev_repr_reduced_m1=ev_repr_reduced_m1,
            images=images,
            max_error=self.MAX_FLOW_ERROR[self._dataset_type])
        wandb_flow_img = wandb.Image(summary_img)
        logger.log_metrics({'train/flow': wandb_flow_img}, step=global_step)

        if 'bezier_prediction' in outputs.keys():
            bezier_prediction: BezierCurves
            bezier_prediction = outputs['bezier_prediction'].detach(cpu=True)
            assert not bezier_prediction.requires_grad

            if images is not None:
                # images[0] is assumed to be the reference image
                background_img = img_torch_to_numpy(images[0])
            else:
                assert ev_repr_reduced is not None
                background_img = ev_repr_reduced_to_img_grayscale(ev_repr_reduced)

            bezier_img = multi_plot_bezier_array(
                bezier_prediction,
                background_img,
                multi_flow_gt= flow_gt if isinstance(flow_gt, list) else [flow_gt], # Could also put [flow_gt] if not list.
                num_t=10,
                x_add_margin=30,
                y_add_margin=30)
            wandb_bezier_img = wandb.Image(bezier_img)
            logger.log_metrics({'train/bezier': wandb_bezier_img}, step=global_step)

    def on_after_backward(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        global_step = trainer.global_step
        if global_step % self.log_every_n_train_steps != 0:
            return
        named_parameters = pl_module.named_parameters()
        figure = get_grad_flow_figure(named_parameters)
        trainer.logger.log_metrics({'train/gradients': figure}, step=global_step)

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.log_only_numbers:
            return
        if not self._training_started:
            # Don't log before the training started.
            # In PL, there is a validation sanity check.
            return
        if self._val_batch_indices is None:
            val_batch_indices = self._set_val_batch_indices()
            self._subsample_val_data(val_batch_indices)

        flow_gt = self._val_data[self.FLOW_GT]
        flow_pred = self._val_data[self.FLOW_PRED]
        flow_valid = self._val_data[self.FLOW_VALID]
        ev_repr_reduced = self._val_data[self.EV_REPR_REDUCED]
        ev_repr_reduced_m1 = self._val_data[self.EV_REPR_REDUCED_M1]
        bezier_params = self._val_data[self.BEZIER_PARAMS]
        images = self._val_data[self.IMAGES]

        if flow_gt:
            assert flow_pred
            # Stack in batch dimension from list for make_grid
            if isinstance(flow_gt[0], list):
                # [[(2, H, W), ...], ...]
                # outer: val batch indices
                # inner: number of gt samples in time
                # We only want the inner loop but batched
                # This is the same as transposing the list -> use pytorch because we can just use their transpose
                # V: number of val samples
                # T: number of gt samples in time

                # V, T, 2, H, W
                new_flow_gt = torch.stack([torch.stack([single_gt_map for single_gt_map in val_sample]) for val_sample in flow_gt])
                # V, T, 2, H, W -> T, V, 2, H, W
                new_flow_gt = torch.transpose(new_flow_gt, 0, 1)
                new_flow_gt = torch.split(new_flow_gt, [1]*new_flow_gt.shape[0], dim=0)
                # [(V, 2, H, W), ... ] T times :) the last item is the predictions for the final prediction
                flow_gt = [x.squeeze() for x in new_flow_gt]
            else:
                assert isinstance(flow_gt[0], torch.Tensor)
                flow_gt = torch.stack(flow_gt)
            flow_pred = torch.stack(flow_pred)
            flow_valid = torch.stack(flow_valid) if len(flow_valid) > 0 else None
            if len(ev_repr_reduced) == 0:
                ev_repr_reduced = None
            else:
                ev_repr_reduced = torch.stack(ev_repr_reduced)
            if len(ev_repr_reduced_m1) == 0:
                ev_repr_reduced_m1 = None
            else:
                ev_repr_reduced_m1 = torch.stack(ev_repr_reduced_m1)
            if len(images) == 0:
                images = None
            else:
                images = [torch.stack([x[0] for x in images]), torch.stack([x[1] for x in images])]

            summary_img = create_summary_img(
                flow_pred,
                flow_gt[-1] if isinstance(flow_gt, list) else flow_gt,
                flow_valid,
                ev_repr_reduced=ev_repr_reduced,
                ev_repr_reduced_m1=ev_repr_reduced_m1,
                images=images,
                max_error=self.MAX_FLOW_ERROR[self._dataset_type])
            wandb_flow_img = wandb.Image(summary_img)

            global_step = trainer.global_step
            logger: WandbLogger = trainer.logger
            logger.log_metrics({'val/flow': wandb_flow_img}, step=global_step)

            if len(bezier_params) > 0:
                bezier_params = torch.stack(bezier_params)
                bezier_curves = BezierCurves(bezier_params)

                if images is not None:
                    # images[0] is assumed to be the reference image
                    background_img = img_torch_to_numpy(images[0])
                else:
                    assert ev_repr_reduced is not None
                    background_img = ev_repr_reduced_to_img_grayscale(ev_repr_reduced)

                bezier_img = multi_plot_bezier_array(
                    bezier_curves,
                    background_img,
                    multi_flow_gt= flow_gt if isinstance(flow_gt, list) else [flow_gt], # Could also put [flow_gt] if not list.
                    num_t=10,
                    x_add_margin=30,
                    y_add_margin=30)
                wandb_bezier_img = wandb.Image(bezier_img)
                logger.log_metrics({'val/bezier': wandb_bezier_img}, step=global_step)

        self._clear_val_data()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.log_only_numbers:
            return
        # NOTE: How to resolve the growing memory issue throughout the validation run?
        #       A hack would be to only return what we want to log in the validation_step function of the LightningModule,
        #       and then log the data in the validation_epoch_end function.
        if not self._training_started:
            # Don't log before the training started.
            # In PL, there is a validation sanity check.
            return
        if self._dataset_type is None:
            self._dataset_type = batch[DataLoading.DATASET_TYPE][0].cpu().item()
        if self._val_batch_indices is not None:
            # NOTE: For the first validation run, we still save everything which can lead to crashes due to full RAM.
            # Once we have set the validation batch indices,
            # we only save data from those.
            if batch_idx not in self._val_batch_indices:
                return

        if isinstance(outputs['gt'], list):
            flow_gt = [x[0].cpu() for x in outputs['gt']]
            # -> list(list(tensors)) will available after end of validation epoch
        else:
            flow_gt = outputs['gt'][0].cpu()
            # -> list(tensors) will available after end of validation epoch
        flow_pred = outputs['pred'][0].cpu()
        flow_valid = outputs['gt_valid'][0].cpu() if 'gt_valid' in outputs else None
        ev_repr_reduced = None
        if 'ev_repr_reduced' in outputs.keys():
            ev_repr_reduced = outputs['ev_repr_reduced'][0].cpu()
        ev_repr_reduced_m1 = None
        if 'ev_repr_reduced_m1' in outputs.keys():
            ev_repr_reduced_m1 = outputs['ev_repr_reduced_m1'][0].cpu()
        images = None
        if 'images' in outputs.keys():
            images = [x[0].detach().cpu() for x in outputs['images']]
            assert len(images) == 2

        bezier_params = None
        if 'bezier_prediction' in outputs.keys():
            bezier_prediction: BezierCurves = outputs['bezier_prediction']
            assert not bezier_prediction.requires_grad
            bezier_params = bezier_prediction.get_params()[0].cpu()

        self._save_val_data(self.FLOW_GT, flow_gt)
        self._save_val_data(self.FLOW_PRED, flow_pred)
        if flow_valid is not None:
            self._save_val_data(self.FLOW_VALID, flow_valid)
        if ev_repr_reduced is not None:
            self._save_val_data(self.EV_REPR_REDUCED, ev_repr_reduced)
        if ev_repr_reduced_m1 is not None:
            self._save_val_data(self.EV_REPR_REDUCED_M1, ev_repr_reduced_m1)
        if bezier_params is not None:
            self._save_val_data(self.BEZIER_PARAMS, bezier_params)
        if images is not None:
            self._save_val_data(self.IMAGES, images)
        self._save_val_data(self.VAL_BATCH_IDX, batch_idx)

    def _set_val_batch_indices(self):
        val_indices = self._val_data[self.VAL_BATCH_IDX]
        assert val_indices
        num_samples = min(len(val_indices), self.log_n_val_predictions)

        if self.deterministic:
            random.seed(0)
        sampled_indices = random.sample(val_indices, num_samples)
        self._val_batch_indices = set(sampled_indices)
        return self._val_batch_indices

    def _clear_val_data(self):
        self._val_data = {
            self.FLOW_GT: list(),
            self.FLOW_PRED: list(),
            self.FLOW_VALID: list(),
            self.EV_REPR_REDUCED: list(),
            self.EV_REPR_REDUCED_M1: list(),
            self.VAL_BATCH_IDX: list(),
            self.BEZIER_PARAMS: list(),
            self.IMAGES: list(),
        }

    def _subsample_val_data(self, val_indices: Union[list, set]):
        for k, v in self._val_data.items():
            if k == self.VAL_BATCH_IDX:
                continue
            assert isinstance(v, list)
            subsampled_list = list()
            for idx, x in enumerate(v):
                if idx not in val_indices:
                    continue
                assert is_cpu(x)
                subsampled_list.append(x)
            self._val_data[k] = subsampled_list
        self._val_data[self.VAL_BATCH_IDX] = list(val_indices)

    def _save_val_data(self, key: str, data: Union[torch.Tensor, int, float, List[torch.Tensor]]):
        assert key in self._val_data.keys()
        if isinstance(data, torch.Tensor) or isinstance(data, list):
            assert is_cpu(data)
        self._val_data[key].append(data)
