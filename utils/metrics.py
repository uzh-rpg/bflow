import math
from typing import List, Optional

import torch as th
from torchmetrics import Metric

from utils.losses import l1_loss_channel_masked


class L1ChannelMasked(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("l1", default=th.tensor(0, dtype=th.float64), dist_reduce_fx="sum")
        self.add_state("total", default=th.tensor(0, dtype=th.int64), dist_reduce_fx="sum")

    def update(self, source: th.Tensor, target: th.Tensor, valid_mask: Optional[th.Tensor]=None):
        # source (prediction): (N, C, *),
        # target (ground truth): (N, C, *), source.shape == target.shape
        # valid_mask: (N, *), where the channel dimension is missing. I.e. valid_mask.ndim == target.ndim - 1

        self.l1 += l1_loss_channel_masked(source, target, valid_mask).double()
        self.total += 1

    def compute(self):
        assert self.total > 0
        return (self.l1 / self.total).float()


class EPE(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("epe", default=th.tensor(0, dtype=th.float64), dist_reduce_fx="sum")
        self.add_state("total", default=th.tensor(0, dtype=th.int64), dist_reduce_fx="sum")

    def update(self, source: th.Tensor, target: th.Tensor, valid_mask: Optional[th.Tensor]=None):
        # source (prediction): (N, C, *),
        # target (ground truth): (N, C, *), source.shape == target.shape
        # valid_mask: (N, *), where the channel dimension is missing. I.e. valid_mask.ndim == target.ndim - 1

        epe = epe_masked(source, target, valid_mask)
        if epe is not None:
            self.epe += epe.double()
            self.total += 1

    def compute(self):
        assert self.total > 0
        return (self.epe / self.total).float()

class EPE_MULTI(Metric):
    def __init__(self, dist_sync_on_step=False, min_traj_len=None, max_traj_len=None):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("epe", default=th.tensor(0, dtype=th.float64), dist_reduce_fx="sum")
        self.add_state("total", default=th.tensor(0, dtype=th.int64), dist_reduce_fx="sum")
        self.min_traj_len = min_traj_len
        self.max_traj_len = max_traj_len

    @staticmethod
    def compute_traj_len(target: List[th.Tensor]):
        target_stack = th.stack(target, dim=0)
        diff = target_stack[1:] - target_stack[:-1]
        return diff.square().sum(dim=2).sqrt().sum(dim=0)

    def get_true_mask(self, target: List[th.Tensor], device: th.device):
        valid_shape = (target[0].shape[0],) + target[0].shape[2:]
        return th.ones(valid_shape, dtype=th.bool, device=device)

    def update(self, source: List[th.Tensor], target: List[th.Tensor], valid_mask: Optional[List[th.Tensor]]=None):
        # source_lst: [(N, C, *), ...], M evaluation/predictions
        # target_lst: [(N, C, *), ...], source_lst[*].shape == target_lst[*].shape
        # valid_mask_lst: [(N, *), ...], where the channel dimension is missing. I.e. valid_mask_lst[*].ndim == target_lst[*].ndim - 1

        if self.min_traj_len is not None or self.max_traj_len is not None:
            traj_len = self.compute_traj_len(target=target)
            valid_len = self.get_true_mask(target=target, device=target[0].device)
            if self.min_traj_len is not None:
                valid_len &= (traj_len >= self.min_traj_len)
            if self.max_traj_len is not None:
                valid_len &= (traj_len <= self.max_traj_len)
            if valid_mask is None:
                valid_mask = [valid_len.clone() for _ in range(len(target))]
            else:
                valid_mask = [valid_mask[idx] & valid_len for idx in range(len(target))]

        epe = epe_masked_multi(source, target, valid_mask)
        if epe is not None:
            self.epe += epe.double()
            self.total += 1

    def compute(self):
        assert self.total > 0
        return (self.epe / self.total).float()

class AE(Metric):
    def __init__(self, degrees: bool=True, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.degrees = degrees

        self.add_state("ae", default=th.tensor(0, dtype=th.float64), dist_reduce_fx="sum")
        self.add_state("total", default=th.tensor(0, dtype=th.int64), dist_reduce_fx="sum")

    def update(self, source: th.Tensor, target: th.Tensor, valid_mask: Optional[th.Tensor]=None):
        # source (prediction): (N, C, *),
        # target (ground truth): (N, C, *), source.shape == target.shape
        # valid_mask: (N, *), where the channel dimension is missing. I.e. valid_mask.ndim == target.ndim - 1

        self.ae += ae_masked(source, target, valid_mask, degrees=self.degrees).double()
        self.total += 1

    def compute(self):
        assert self.total > 0
        return (self.ae / self.total).float()


class AE_MULTI(Metric):
    def __init__(self, degrees: bool=True, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.degrees = degrees

        self.add_state("ae", default=th.tensor(0, dtype=th.float64), dist_reduce_fx="sum")
        self.add_state("total", default=th.tensor(0, dtype=th.int64), dist_reduce_fx="sum")

    def update(self, source: List[th.Tensor], target: List[th.Tensor], valid_mask: Optional[List[th.Tensor]]=None):
        # source_lst: [(N, C, *), ...], M evaluation/predictions
        # target_lst: [(N, C, *), ...], source_lst[*].shape == target_lst[*].shape
        # valid_mask_lst: [(N, *), ...], where the channel dimension is missing. I.e. valid_mask_lst[*].ndim == target_lst[*].ndim - 1

        self.ae += ae_masked_multi(source, target, valid_mask, degrees=self.degrees).double()
        self.total += 1

    def compute(self):
        assert self.total > 0
        return (self.ae / self.total).float()

class NPE(Metric):
    def __init__(self, n_pixels: float, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        assert n_pixels > 0
        self.n_pixels = n_pixels

        self.add_state("npe", default=th.tensor(0, dtype=th.float64), dist_reduce_fx="sum")
        self.add_state("total", default=th.tensor(0, dtype=th.int64), dist_reduce_fx="sum")

    def update(self, source: th.Tensor, target: th.Tensor, valid_mask: Optional[th.Tensor]=None):
        # source (prediction): (N, C, *),
        # target (ground truth): (N, C, *), source.shape == target.shape
        # valid_mask: (N, *), where the channel dimension is missing. I.e. valid_mask.ndim == target.ndim - 1

        self.npe += n_pixel_error_masked(source, target, valid_mask, self.n_pixels).double()
        self.total += 1

    def compute(self):
        assert self.total > 0
        return (self.npe / self.total).float()

def n_pixel_error_masked(source: th.Tensor, target: th.Tensor, valid_mask: Optional[th.Tensor], n_pixels: float):
    # source: (N, C, *),
    # target: (N, C, *), source.shape == target.shape
    # valid_mask: (N, *), where the channel dimension is missing. I.e. valid_mask.ndim == target.ndim - 1
    assert source.ndim > 2
    assert source.shape == target.shape

    if valid_mask is not None:
        assert valid_mask.shape[0] == target.shape[0]
        assert valid_mask.ndim == target.ndim - 1
        assert valid_mask.dtype == th.bool

        num_valid = th.sum(valid_mask)
        assert num_valid > 0

    gt_flow_magn = th.linalg.norm(target, dim=1)
    error_magn = th.linalg.norm(source - target, dim=1)

    if valid_mask is not None:
        rel_error = th.zeros_like(error_magn)
        rel_error[valid_mask] = error_magn[valid_mask] / th.clip(gt_flow_magn[valid_mask], min=1e-6)
    else:
        rel_error = error_magn / th.clip(gt_flow_magn, min=1e-6)

    error_map = (error_magn > n_pixels) & (rel_error >= 0.05)

    if valid_mask is not None:
        error = error_map[valid_mask].sum() / num_valid
    else:
        error = th.mean(error_map.float())

    error *= 100
    return error


def epe_masked(source: th.Tensor, target: th.Tensor, valid_mask: Optional[th.Tensor] = None) -> Optional[th.Tensor]:
    # source: (N, C, *),
    # target: (N, C, *), source.shape == target.shape
    # valid_mask: (N, *), where the channel dimension is missing. I.e. valid_mask.ndim == target.ndim - 1
    assert source.ndim > 2
    assert source.shape == target.shape

    epe = th.sqrt(th.square(source - target).sum(1))
    if valid_mask is not None:
        assert valid_mask.shape[0] == target.shape[0]
        assert valid_mask.ndim == target.ndim - 1
        assert valid_mask.dtype == th.bool
        assert epe.shape == valid_mask.shape
        denominator = valid_mask.sum()
        if denominator == 0:
            return None
        return epe[valid_mask].sum() / denominator
    return th.mean(epe)


def epe_masked_multi(source_lst: List[th.Tensor], target_lst: List[th.Tensor], valid_mask_lst: Optional[List[th.Tensor]] = None) -> Optional[th.Tensor]:
    # source_lst: [(N, C, *), ...], M evaluation/predictions
    # target_lst: [(N, C, *), ...], source_lst[*].shape == target_lst[*].shape
    # valid_mask_lst: [(N, *), ...], where the channel dimension is missing. I.e. valid_mask_lst[*].ndim == target_lst[*].ndim - 1

    num_preds = len(source_lst)
    assert num_preds > 0
    assert len(target_lst) == num_preds, len(target_lst)
    if valid_mask_lst is not None:
        assert len(valid_mask_lst) == num_preds, len(valid_mask_lst)
    else:
        valid_mask_lst = [None]*num_preds
    epe_sum = 0
    denominator = 0
    for source, target, valid_mask in zip(source_lst, target_lst, valid_mask_lst):
        epe = epe_masked(source, target, valid_mask)
        if epe is not None:
            epe_sum += epe
            denominator += 1
    if denominator == 0:
        return None
    epe_sum: th.Tensor
    return epe_sum / denominator

def ae_masked_multi(source_lst: List[th.Tensor], target_lst: List[th.Tensor], valid_mask_lst: Optional[List[th.Tensor]]=None, degrees: bool=True) -> th.Tensor:
    # source_lst: [(N, C, *), ...], M evaluation/predictions
    # target_lst: [(N, C, *), ...], source_lst[*].shape == target_lst[*].shape
    # valid_mask_lst: [(N, *), ...], where the channel dimension is missing. I.e. valid_mask_lst[*].ndim == target_lst[*].ndim - 1

    num_preds = len(source_lst)
    assert num_preds > 0
    assert len(target_lst) == num_preds, len(target_lst)
    if valid_mask_lst is not None:
        assert len(valid_mask_lst) == num_preds, len(valid_mask_lst)
    else:
        valid_mask_lst = [None]*num_preds
    ae_sum = 0
    for source, target, valid_mask in zip(source_lst, target_lst, valid_mask_lst):
        ae_sum += ae_masked(source, target, valid_mask, degrees)
    ae_sum: th.Tensor
    return ae_sum / num_preds


def ae_masked(source: th.Tensor, target: th.Tensor, valid_mask: Optional[th.Tensor]=None, degrees: bool=True) -> th.Tensor:
    # source: (N, C, *),
    # target: (N, C, *), source.shape == target.shape
    # valid_mask: (N, *), where the channel dimension is missing. I.e. valid_mask.ndim == target.ndim - 1
    assert source.ndim > 2
    assert source.shape == target.shape

    shape = list(source.shape)
    extension_shape = shape
    extension_shape[1] = 1
    extension = th.ones(extension_shape, device=source.device)

    source_ext = th.cat((source, extension), dim=1)
    target_ext = th.cat((target, extension), dim=1)

    # according to https://vision.middlebury.edu/flow/floweval-ijcv2011.pdf

    nominator = th.sum(source_ext * target_ext, dim=1)
    denominator = th.linalg.norm(source_ext, dim=1) * th.linalg.norm(target_ext, dim=1)

    tmp = th.div(nominator, denominator)

    # Somehow this seems necessary
    tmp[tmp > 1.0] = 1.0
    tmp[tmp < -1.0] = -1.0

    ae = th.acos(tmp)
    if degrees:
        ae = ae/math.pi*180

    if valid_mask is not None:
        assert valid_mask.shape[0] == target.shape[0]
        assert valid_mask.ndim == target.ndim - 1
        assert valid_mask.dtype == th.bool
        assert ae.shape == valid_mask.shape
        ae_masked = ae[valid_mask].sum() / valid_mask.sum()
        return ae_masked
    return th.mean(ae)

def predictions_from_lin_assumption(source: th.Tensor, target_timestamps: List[float]) -> List[th.Tensor]:
    assert max(target_timestamps) <= 1
    assert 0 <= min(target_timestamps)

    output = list()
    for target_ts in target_timestamps:
        output.append(target_ts * source)
    return output
