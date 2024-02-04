from typing import List, Optional

import torch as th


def l1_loss_channel_masked(source: th.Tensor, target: th.Tensor, valid_mask: Optional[th.Tensor]=None):
    # source: (N, C, *)
    # target: (N, C, *), source.shape == target.shape
    # valid_mask: (N, *), where the channel dimension is missing. I.e. valid_mask.ndim == target.ndim - 1
    assert source.ndim > 2
    assert source.shape == target.shape

    loss = th.abs(source - target).sum(1)
    if valid_mask is not None:
        assert valid_mask.shape[0] == target.shape[0]
        assert valid_mask.ndim == target.ndim - 1
        assert valid_mask.dtype == th.bool
        assert loss.shape == valid_mask.shape
        loss_masked = loss[valid_mask].sum() / valid_mask.sum()
        return loss_masked
    return th.mean(loss)


def l1_seq_loss_channel_masked(source_list: List[th.Tensor], target: th.Tensor, valid_mask: Optional[th.Tensor]=None, gamma: float=0.8):
    # source: [(N, C, *), ...], I predictions from I iterations
    # target: (N, C, *), source.shape == target.shape
    # valid_mask: (N, *), where the channel dimension is missing. I.e. valid_mask.ndim == target.ndim - 1

    # Adopted from https://github.com/princeton-vl/RAFT/blob/224320502d66c356d88e6c712f38129e60661e80/train.py#L47

    n_predictions = len(source_list)
    loss = 0

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = l1_loss_channel_masked(source_list[i], target, valid_mask)
        loss += i_weight * i_loss

    return loss

def l1_multi_seq_loss_channel_masked(src_list_list: List[List[th.Tensor]], target_list: List[th.Tensor], valid_mask_list: Optional[List[th.Tensor]]=None, gamma: float=0.8):
    # src_list_list: [[(N, C, *), ...], ...], I*M predictions -> I iterations (outer) and M supervision targets (inner)
    # target_list: [(N, C, *), ...], M supervision targets
    # valid_mask_list: [(N, *), ...], M supervision targets

    loss = 0

    num_iters = len(src_list_list)
    for iter_idx, sources_per_iter in enumerate(src_list_list):
        # iteration over RAFT iterations
        num_targets = len(sources_per_iter)
        assert num_targets > 0
        assert num_targets == len(target_list)
        i_loss = 0
        for tindex, source in enumerate(sources_per_iter):
            # iteration over prediction times
            i_loss += l1_loss_channel_masked(source, target_list[tindex], valid_mask_list[tindex] if valid_mask_list is not None else None)

        i_loss /= num_targets
        i_weight = gamma**(num_iters - iter_idx - 1)
        loss += i_weight * i_loss

    return loss