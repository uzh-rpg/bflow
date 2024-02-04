from typing import List, Dict, Any, Union

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torchmetrics import MetricCollection

from data.utils.keys import DataLoading, DataSetType
from models.raft_spline.raft import RAFTSpline, BezierCurves
from modules.utils import detach_tensors, reduce_ev_repr, InputPadder
from utils.general import to_cpu
from utils.losses import l1_seq_loss_channel_masked, l1_multi_seq_loss_channel_masked
from utils.metrics import EPE, AE, NPE, EPE_MULTI, AE_MULTI, predictions_from_lin_assumption


class RAFTSplineModule(pl.LightningModule):
    def __init__(self, config: Union[Dict[str, Any], DictConfig]):
        super().__init__()

        self.num_iter_train = config['model']['num_iter']['train']
        self.num_iter_test = config['model']['num_iter']['test']

        self._input_padder = InputPadder(min_size=8, no_top_padding=False)
        self.net = RAFTSpline(config['model'])

        self.use_images = config['model']['use_boundary_images']
        self.use_events = config['model']['use_events']

        self.train_params = config['training']
        self.train_with_multi_loss = self.train_params['multi_loss']

        single_metrics = MetricCollection({
            'epe': EPE(),
            'ae': AE(degrees=True),
            '1pe': NPE(1),
            '2pe': NPE(2),
            '3pe': NPE(3),
        })

        multi_metrics = MetricCollection({
            'epe_multi': EPE_MULTI(),
            'ae_multi': AE_MULTI(degrees=True),
        })

        self.train_single_metrics = single_metrics.clone(prefix='train/')
        self.train_multi_metrics = multi_metrics.clone(prefix='train/')

        self.val_single_metrics = single_metrics.clone(prefix='val/')
        self.val_multi_metrics = multi_metrics.clone(prefix='val/')

        # To evaluate a pseudo-linear prediction with the multi metrics:
        self.train_epe_multi_lin = EPE_MULTI()
        self.train_ae_multi_lin = AE_MULTI(degrees=True)
        self.val_epe_multi_lin = EPE_MULTI()
        self.val_ae_multi_lin = AE_MULTI(degrees=True)

    def forward(self, voxel_grid, images, iters, test_mode: bool):
        return self.net(voxel_grid=voxel_grid, images=images, iters=iters, test_mode=test_mode)

    '''
    We detach here (without moving to cpu) because PL is throwing a deprecated warning that this should be done for version >= 1.6.
    '''
    @detach_tensors(cpu=False)
    def training_step(self, batch, batch_idx):
        # forward_flow: (N, 2, H, W), float32
        forward_flow_gt = batch[DataLoading.FLOW]
        # forward_flow_valid: (N, H, W), bool
        forward_flow_gt_valid = batch[DataLoading.FLOW_VALID] if DataLoading.FLOW_VALID in batch else None
        # event_representation (DSEC): (N, 2*num_bins-1, H, W), float32
        # event_representation (MULTIFLOW2D): (N, nbins_context + nbins_corr - 1, H, W), float32
        ev_repr = batch[DataLoading.EV_REPR]

        self.log('global_step', self.trainer.global_step)

        if self.use_images:
            images = batch[DataLoading.IMG]
        else:
            images = None

        if self._input_padder.requires_padding(ev_repr):
            # You can implement it here or adapt the crop size to avoid padding at all during training
            raise NotImplementedError

        dataset_type = batch[DataLoading.DATASET_TYPE][0]

        output = dict()
        if dataset_type == DataSetType.DSEC:
            # NOTE: We just do this to extract voxel grids that belong to start and end time of the forward optical flow
            combined_bins = ev_repr.shape[1]
            assert combined_bins % 2 == 1
            num_bins = combined_bins // 2 + 1
            assert num_bins >= 1
            ev_repr_previous = ev_repr[:, 0:num_bins, ...]
            ev_repr_current = ev_repr[:, -num_bins:, ...]

            # bezier_up_predictions: list(BezierQuadratic, ...), float32 or float16
            bezier_up_predictions: List[BezierCurves] = self(
                voxel_grid=ev_repr if self.use_events else None,
                images=images,
                iters=self.num_iter_train,
                test_mode=False)
            # forward_flow_preds: list((N, 2, 480, 640), ...), float32 or float16
            forward_flow_preds = [x.get_flow_from_reference(1.0) for x in bezier_up_predictions]
            l1_seq_loss = l1_seq_loss_channel_masked(forward_flow_preds, forward_flow_gt, forward_flow_gt_valid)

            self.log('train/l1_seq_loss', l1_seq_loss, logger=True, on_epoch=True)

            train_metrics = self.train_single_metrics(forward_flow_preds[-1], forward_flow_gt, forward_flow_gt_valid)
            self.log_dict(train_metrics, logger=True, on_epoch=True)

            output.update({
                'loss': l1_seq_loss, # required
                'pred': forward_flow_preds[-1],
                'gt': forward_flow_gt,
                'gt_valid': forward_flow_gt_valid,
            })
        elif dataset_type == DataSetType.MULTIFLOW2D:
            nbins_context = batch[DataLoading.BIN_META]['nbins_context'][0]
            nbins_corr = batch[DataLoading.BIN_META]['nbins_correlation'][0]
            nbins_total = ev_repr.shape[1]
            assert nbins_total == batch[DataLoading.BIN_META]['nbins_total'][0] == nbins_context + nbins_corr - 1
            ev_repr_previous = ev_repr[:, 0:nbins_corr, ...]
            ev_repr_current = ev_repr[:, -nbins_corr:, ...]

            forward_flow_ts = batch[DataLoading.FLOW_TIMESTAMPS]

            # only use boundary images
            bezier_up_predictions: List[BezierCurves] = self(
                voxel_grid=ev_repr if self.use_events else None,
                images=images,
                iters=self.num_iter_train,
                test_mode=False)

            timestamp_eval_lst = list()
            for timestamp_batch in forward_flow_ts:
                # HACK: assume that timestamps are essentially the same along the batch dim
                ts_mean_diff = (timestamp_batch[1:] - timestamp_batch[:-1]).abs().mean().item()
                assert 0 <= ts_mean_diff < 0.001
                timestamp_eval_lst.append(timestamp_batch[0].item())
            # outer list -> iteration, inner list -> eval times
            forward_flow_preds = list()
            for bezier_up_iter in bezier_up_predictions:
                flow_preds_per_iter = list()
                for timestamp in timestamp_eval_lst:
                    flow_preds_per_iter.append(bezier_up_iter.get_flow_from_reference(timestamp))
                forward_flow_preds.append(flow_preds_per_iter)

            if self.train_with_multi_loss:
                loss = l1_multi_seq_loss_channel_masked(forward_flow_preds, forward_flow_gt)
                self.log('train/l1_multi_seq_loss', loss, logger=True, on_epoch=True)
            else:
                loss = l1_seq_loss_channel_masked(forward_flow_preds[-1], forward_flow_gt[-1])
                self.log('train/l1_seq_loss', loss, logger=True, on_epoch=True)

            train_single_metrics = self.train_single_metrics(forward_flow_preds[-1][-1], forward_flow_gt[-1])
            self.log_dict(train_single_metrics, logger=True, on_epoch=True)
            train_multi_metrics = self.train_multi_metrics(forward_flow_preds[-1], forward_flow_gt)
            self.log_dict(train_multi_metrics, logger=True, on_epoch=True)

            # evaluate against linearity assumption (both space and time)
            forward_flow_preds_lin = predictions_from_lin_assumption(forward_flow_preds[-1][-1], timestamp_eval_lst)
            epe_multi_lin = self.train_epe_multi_lin(forward_flow_preds_lin, forward_flow_gt)
            ae_multi_lin = self.train_ae_multi_lin(forward_flow_preds_lin, forward_flow_gt)

            self.log('train/epe_multi_lin', epe_multi_lin, logger=True, on_epoch=True)
            self.log('train/ae_multi_lin', ae_multi_lin, logger=True, on_epoch=True)

            output.update({
                'loss': loss, # required
                'pred': forward_flow_preds[-1][-1],
                'gt': forward_flow_gt, # list of gt -> [(N, 2, H, W), ...] (M times)
                #'gt': forward_flow_gt[-1],
            })
        else:
            raise NotImplementedError

        output.update({
            'bezier_prediction': bezier_up_predictions[-1].detach(),
        })
        if self.use_events:
            output.update({
                'ev_repr_reduced': reduce_ev_repr(ev_repr_current),
                'ev_repr_reduced_m1': reduce_ev_repr(ev_repr_previous),
            })
        if images is not None:
            output.update({'images': images})

        return output

    @to_cpu
    def validation_step(self, batch, batch_idx):
        # forward_flow: (N, 2, 480, 640), float32
        forward_flow_gt = batch[DataLoading.FLOW]
        # forward_flow_valid: (N, 480, 640), bool
        forward_flow_gt_valid = batch[DataLoading.FLOW_VALID] if DataLoading.FLOW_VALID in batch else None
        # event_representation: (N, 2*num_bins-1, 480, 640), float32
        ev_repr = batch[DataLoading.EV_REPR]
        if self.use_images:
            images = batch[DataLoading.IMG]
        else:
            images = None

        dataset_type = batch[DataLoading.DATASET_TYPE][0]

        output = dict()

        if dataset_type == DataSetType.DSEC:
            # NOTE: We just do this to extract voxel grids that belong to start and end time of the forward optical flow
            combined_bins = ev_repr.shape[1]
            assert combined_bins % 2 == 1, f'combined_bins={combined_bins}'
            num_bins = combined_bins // 2 + 1
            assert num_bins >= 1
            ev_repr_previous = ev_repr[:, 0:num_bins, ...]
            ev_repr_current = ev_repr[:, -num_bins:, ...]

            requires_padding = self._input_padder.requires_padding(ev_repr)
            if requires_padding:
                ev_repr = self._input_padder.pad(ev_repr)
                if images is not None:
                    assert len(images) == 2
                    images = [self._input_padder.pad(x) for x in images]

            bezier_low, bezier_up = self(
                voxel_grid=ev_repr if self.use_events else None,
                images=images,
                iters=self.num_iter_test,
                test_mode=True)
            bezier_up: BezierCurves
            # forward_flow_pred: (N, 2, 480, 640), float32 or float16
            forward_flow_pred = bezier_up.get_flow_from_reference(1.0)

            if requires_padding:
                forward_flow_pred = self._input_padder.unpad(forward_flow_pred)
                if images is not None:
                    images = [self._input_padder.unpad(x) for x in images]

            val_metrics = self.val_single_metrics(forward_flow_pred, forward_flow_gt, forward_flow_gt_valid)
            self.log_dict(val_metrics, logger=True, on_epoch=True)

            output.update({
                'pred': forward_flow_pred,
                'gt': forward_flow_gt,
                'gt_valid': forward_flow_gt_valid,
            })
        elif dataset_type == DataSetType.MULTIFLOW2D:
            nbins_context = batch[DataLoading.BIN_META]['nbins_context'][0]
            nbins_corr = batch[DataLoading.BIN_META]['nbins_correlation'][0]
            nbins_total = ev_repr.shape[1]
            assert nbins_total == batch[DataLoading.BIN_META]['nbins_total'][0] == nbins_context + nbins_corr - 1
            ev_repr_previous = ev_repr[:, 0:nbins_corr, ...]
            ev_repr_current = ev_repr[:, -nbins_corr:, ...]

            forward_flow_ts = batch[DataLoading.FLOW_TIMESTAMPS]

            requires_padding = self._input_padder.requires_padding(ev_repr)
            if requires_padding:
                ev_repr = self._input_padder.pad(ev_repr)
                ev_repr_previous = self._input_padder.pad(ev_repr_previous)
                ev_repr_current = self._input_padder.pad(ev_repr_current)
                if images is not None:
                    images = [self._input_padder.pad(x) for x in images]

            bezier_low, bezier_up = self(
                voxel_grid=ev_repr if self.use_events else None,
                images=images,
                iters=self.num_iter_test,
                test_mode=True)
            bezier_up: BezierCurves

            forward_flow_preds = list()
            timestamp_eval_lst = list()
            for timestamp_batch in forward_flow_ts:
                # HACK: check that timestamps are essentially the same
                ts_mean_diff = (timestamp_batch[1:] - timestamp_batch[:-1]).abs().mean().item()
                assert 0 <= ts_mean_diff < 0.001, ts_mean_diff
                timestamp = timestamp_batch[0].item()

                timestamp_eval_lst.append(timestamp)

                forward_flow_pred = bezier_up.get_flow_from_reference(timestamp)
                if requires_padding:
                    forward_flow_pred = self._input_padder.unpad(forward_flow_pred)
                forward_flow_preds.append(forward_flow_pred)
            if requires_padding and images is not None:
                images = [self._input_padder.unpad(x) for x in images]

            val_single_metrics = self.val_single_metrics(forward_flow_preds[-1], forward_flow_gt[-1])
            self.log_dict(val_single_metrics, logger=True, on_epoch=True)
            val_multi_metrics = self.val_multi_metrics(forward_flow_preds, forward_flow_gt)
            self.log_dict(val_multi_metrics, logger=True, on_epoch=True)

            # evaluate against linearity assumption (both space and time)
            forward_flow_preds_lin = predictions_from_lin_assumption(forward_flow_preds[-1], timestamp_eval_lst)
            epe_multi_lin = self.val_epe_multi_lin(forward_flow_preds_lin, forward_flow_gt)
            ae_multi_lin = self.val_ae_multi_lin(forward_flow_preds_lin, forward_flow_gt)

            self.log('val/epe_multi_lin', epe_multi_lin, logger=True)
            self.log('val/ae_multi_lin', ae_multi_lin, logger=True)

            output.update({
                'pred': forward_flow_preds[-1],
                'gt': forward_flow_gt, # list of gt -> [(N, 2, H, W), ...] (M times)
                #'gt': forward_flow_gt[-1],
            })
        else:
            raise NotImplementedError

        output.update({
            'bezier_prediction': bezier_up,
        })
        if self.use_events:
            output.update({
                'ev_repr_reduced': reduce_ev_repr(ev_repr_current),
                'ev_repr_reduced_m1': reduce_ev_repr(ev_repr_previous),
            })
        if images is not None:
            output.update({'images': images})

        return output

    def configure_optimizers(self):
        lr = self.train_params['learning_rate']
        weight_decay = self.train_params['weight_decay']
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

        scheduler_params = self.train_params['lr_scheduler']
        if not scheduler_params['use']:
            return optimizer

        total_steps = scheduler_params['total_steps']
        assert total_steps is not None
        assert total_steps > 0
        pct_start = scheduler_params['pct_start']
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=lr,
            total_steps=total_steps+100,
            pct_start=pct_start,
            cycle_momentum=False,
            anneal_strategy='linear')
        lr_scheduler_config = {
            # REQUIRED: The scheduler instance
            "scheduler": lr_scheduler,
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "step",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            # If set to `True`, will enforce that the value specified 'monitor'
            # is available when the scheduler is updated, thus stopping
            # training if not found. If set to `False`, it will only produce a warning
            "strict": True,
            # If using the `LearningRateMonitor` callback to monitor the
            # learning rate progress, this keyword can be used to specify
            # a custom logged name
            "name": 'learning_rate',
        }

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}
