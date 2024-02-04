from __future__ import annotations

from typing import Union, List, Tuple, Optional

import torch
import torch as th
import torch.nn.functional as F
from omegaconf import ListConfig

from models.raft_utils.utils import bilinear_sampler


class CorrData:
    def __init__(self,
                 corr: th.Tensor,
                 batch_size: int):
        assert isinstance(corr, th.Tensor)
        num_targets, bhw, dim, ht, wd = corr.shape
        assert dim == 1
        assert isinstance(batch_size, int)
        assert batch_size >= 1

        # corr: num_targets, batch_size*ht*wd, 1, ht, wd
        self._corr = corr
        self._batch_size = batch_size
        self._target_indices = None

    @property
    def corr(self) -> th.Tensor:
        assert self._corr.ndim == 5
        return self._corr

    @property
    def corr_batched(self) -> th.Tensor:
        num_targets, bhw, dim, ht, wd = self.corr.shape
        assert dim == 1
        return self.corr.view(-1, dim, ht, wd)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self) -> th.device:
        return self.corr.device

    @property
    def target_indices(self) -> th.Tensor:
        assert self._target_indices is not None
        return self._target_indices

    @target_indices.setter
    def target_indices(self, values: Union[List[int], th.Tensor]):
        if isinstance(values, list):
            values = sorted(values)
            values = th.Tensor(values, device=self.device)
        else:
            assert values.device == self.device
        assert values.dtype == th.int64
        assert values.ndim == 1
        assert values.numel() > 0
        assert values.min() >= 0
        assert th.all(values[1:] - values[:-1] > 0)
        assert self.corr.shape[0] == values.numel()
        self._target_indices = values

    def init_target_indices(self):
        num_targets = self.corr.shape[0]
        self.target_indices = torch.arange(num_targets, device=self.device)

    @staticmethod
    def _extract_database_indices_from_query_matches(query_values: torch.Tensor, database_values: torch.Tensor) -> th.Tensor:
        '''
        query_values: set of unique integers (assumed in ascending order)
        database_values: set of unique integers (assumed in ascending order)

        This function returns the database indices where query values match the database values.
        Example:
            query_values: torch.tensor([4, 5, 9])
            database_values: torch.tensor([1, 4, 5, 6, 9])
            returns: torch.tensor([1, 2, 4])
        '''
        assert database_values.dtype == torch.int64
        assert query_values.dtype == torch.int64
        assert database_values.ndim == 1
        assert query_values.ndim == 1
        assert torch.equal(database_values, torch.unique_consecutive(database_values))
        assert torch.equal(query_values, torch.unique_consecutive(query_values))
        device = query_values.device
        assert device == database_values.device

        num_db = database_values.numel()
        num_q = query_values.numel()

        database_values_expanded = database_values.expand(num_q, -1)
        query_values_expanded = query_values.expand(num_db, -1).transpose(0, 1)

        compare = torch.eq(database_values_expanded, query_values_expanded)

        indices = torch.arange(num_db, device=device)
        indices = indices.expand(num_q, -1)
        assert torch.all(compare.sum(1) == 1), f'compare =\n{compare}'
        # If the previous assertion fails it likely is the case that the query values are not a subset of database values.
        out = indices[compare]
        assert torch.equal(query_values, database_values[out])
        return out

    def get_downsampled(self, target_indices: Union[List[int], th.Tensor]) -> CorrData:
        if isinstance(target_indices, list):
            target_indices = sorted(target_indices)
            target_indices = th.tensor(target_indices, device=self.device)

        indices_to_select = self._extract_database_indices_from_query_matches(target_indices, self.target_indices)
        corr_selected = th.index_select(self.corr, dim=0, index=indices_to_select)

        num_new_targets, bhw, dim, ht, wd = corr_selected.shape
        assert dim == 1
        corr_selected = corr_selected.reshape(-1, dim, ht, wd)
        corr_down = F.avg_pool2d(corr_selected, 2, stride=2)
        _, _, ht_down, wd_down = corr_down.shape
        corr_down = corr_down.view(num_new_targets, bhw, dim, ht_down, wd_down)

        out = CorrData(corr_down, self.batch_size)
        out.target_indices = target_indices
        return out

class CorrComputation:
    def __init__(self,
                 fmap1: Union[th.Tensor, List[th.Tensor]],
                 fmap2: Union[th.Tensor, List[th.Tensor]],
                 num_levels_per_target: Union[int, List[int], List[th.Tensor]]):
        '''
        fmap1:  batch, dim, ht, wd  OR
                List[tensor(batch, dim, ht, wd)]
        fmap2:  batch, dim, ht, wd  OR
                num_targets, batch, dim, ht, wd  OR
                List[tensor(num_targets, batch, dim, ht, wd)]
        -> in case of list input, len(fmap1) == len(fmap2) is enforced
        num_levels_per_target:  int  OR
                                List[int] SUCH THAT len(...) == num_targets  OR
                                List[tensor(num_targets)]
        '''
        self._has_single_reference = isinstance(fmap1, th.Tensor)
        if isinstance(num_levels_per_target, int):
            num_levels_per_target = [num_levels_per_target]
        else:
            assert isinstance(num_levels_per_target, list) or isinstance(num_levels_per_target, ListConfig)
        if self._has_single_reference:
            assert fmap1.ndim == 4
            assert isinstance(fmap2, th.Tensor)
            if fmap2.ndim == 4:
                # This is the case where we also only have a single target
                fmap2 = fmap2.unsqueeze(0)
            else:
                assert fmap2.ndim == 5
            assert fmap1.shape == fmap2.shape[1:]
            fmap1 = [fmap1]
            fmap2 = [fmap2]
            assert len(num_levels_per_target) == fmap2[0].shape[0]
            for x in num_levels_per_target:
                assert isinstance(x, int)
            num_levels_per_target = [th.tensor(num_levels_per_target)]
        else:
            assert isinstance(fmap1, list)
            assert isinstance(fmap2, list)
            assert len(fmap1) == len(fmap2)
            assert len(num_levels_per_target) == len(fmap2)
            for f1, f2, num_lvls in zip(fmap1, fmap2, num_levels_per_target):
                assert isinstance(f1, th.Tensor)
                assert f1.ndim == 4
                assert isinstance(f2, th.Tensor)
                assert f2.ndim == 5
                assert f1.shape == f2.shape[1:]
                assert isinstance(num_lvls, th.Tensor)
                assert num_lvls.dtype == th.int64
                assert num_lvls.numel() == f2.shape[0]
        self._fmap1 = fmap1
        self._fmap2 = fmap2

        self._num_targets_per_reference = [x.shape[0] for x in fmap2]
        self._num_targets_overall = sum(self._num_targets_per_reference)
        assert self._num_targets_overall == th.cat(num_levels_per_target).numel()
        self._num_levels_per_target = num_levels_per_target

        self._bdhw = fmap1[0].shape

    @property
    def batch(self) -> int:
        return self._bdhw[0]

    @property
    def dim(self) -> int:
        return self._bdhw[1]

    @property
    def height(self) -> int:
        return self._bdhw[2]

    @property
    def width(self) -> int:
        return self._bdhw[3]

    @property
    def num_references(self) -> int:
        return len(self._fmap1)

    @property
    def num_levels_per_target(self) -> List[th.Tensor]:
        return self._num_levels_per_target

    @property
    def num_levels_per_target_merged(self) -> th.Tensor:
        return th.cat(self._num_levels_per_target)

    @property
    def num_targets_per_reference(self) -> List[int]:
        return self._num_targets_per_reference

    @property
    def num_targets_overall(self) -> int:
        return self._num_targets_overall

    def __add__(self, other: CorrComputation) -> CorrComputation:
        fmap1 = self._fmap1 + other._fmap1
        fmap2 = self._fmap2 + other._fmap2
        num_levels_per_target = self._num_levels_per_target + other._num_levels_per_target
        return CorrComputation(fmap1=fmap1, fmap2=fmap2, num_levels_per_target=num_levels_per_target)

    def get_correlation_volume(self) -> CorrData:
        # Note: we could also just use the more general N to N code but this should be slightly faster and easier to understand.
        if self._has_single_reference:
            # case: 1 to many, which includes the 1 to 1 special case
            return self._corr_dot_prod_1_to_N()
        # case: many to many
        return self._corr_dot_prod_M_to_N()

    def _corr_dot_prod_1_to_N(self) -> CorrData:
        # 1 to 1 if num_targets_sum is 1, which is a special case of this code.
        assert len(self._fmap1) == 1
        assert len(self._fmap2) == 1
        num_targets_sum = self.num_targets_overall
        fmap1 = self._fmap1[0].view(self.batch, self.dim, self.height*self.width)
        fmap2 = self._fmap2[0].view(num_targets_sum, self.batch, self.dim, self.height*self.width)

        corr_data = self._corr_dot_prod_util(fmap1, fmap2)
        return corr_data

    def _corr_dot_prod_M_to_N(self) -> CorrData:
        assert len(self._fmap1) > 1
        assert len(self._fmap2) > 1
        assert len(self._fmap1) == len(self._fmap2)

        # fmap{1,2}: num_targets_overall, batch, dim, height, width
        fmap1 = th.cat([x.expand(num_trgts, -1, -1, -1, -1) for x, num_trgts in zip(self._fmap1, self.num_targets_per_reference)], dim=0)
        fmap2 = th.cat(self._fmap2, dim=0)

        num_targets_overall = self.num_targets_overall
        fmap1 = fmap1.view(num_targets_overall, self.batch, self.dim, self.height*self.width)
        fmap2 = fmap2.view(num_targets_overall, self.batch, self.dim, self.height*self.width)

        corr_data = self._corr_dot_prod_util(fmap1, fmap2)
        return corr_data

    def _corr_dot_prod_util(self, fmap1: th.Tensor, fmap2: th.Tensor) -> CorrData:
        # corr: num_targets_sum, batch, ht*wd, ht*wd
        corr = fmap1.transpose(-1, -2) @ fmap2
        corr = corr / th.sqrt(th.tensor(self.dim, device=corr.device).float())
        corr = corr.view(self.num_targets_overall, self.batch * self.height * self.width, 1, self.height, self.width)

        out = CorrData(corr=corr, batch_size=self.batch)
        out.init_target_indices()
        return out


class CorrBlockParallelMultiTarget:
    def __init__(self,
                 corr_computation_events: Optional[CorrComputation]=None,
                 corr_computation_frames: Optional[CorrComputation]=None,
                 radius: int=4):
        do_events = corr_computation_events is not None
        do_frames = corr_computation_frames is not None
        assert do_events or do_frames
        assert radius >= 1

        if do_events and not do_frames:
            corr_computation = corr_computation_events
        elif do_frames and not do_events:
            corr_computation = corr_computation_frames
        else:
            assert do_events and do_frames
            corr_computation = corr_computation_events + corr_computation_frames

        num_levels_per_target = corr_computation.num_levels_per_target_merged.tolist()
        self._num_targets_base = len(num_levels_per_target)
        self._radius = radius

        corr_base_data = corr_computation.get_correlation_volume()

        max_num_levels = max(num_levels_per_target)

        self._corr_pyramid = [corr_base_data]
        for num_levels in range(2, max_num_levels + 1):
            target_idx_list = [idx for idx, val in enumerate(num_levels_per_target) if val >= num_levels]
            corr_data = self._corr_pyramid[-1].get_downsampled(target_indices=target_idx_list)
            self._corr_pyramid.append(corr_data)

    def __call__(self, coords: Union[th.Tensor, List[th.Tensor], Tuple[th.Tensor]]) -> th.Tensor:
        if isinstance(coords, list) or isinstance(coords, tuple):
            # num_targets_base, N, 2, H, W
            coords = th.stack(coords, dim=0)
        assert coords.ndim == 5
        # num_targets_base, N, H, W, 2
        coords = coords.permute(0, 1, 3, 4, 2)
        num_targets_base, batch, h1, w1, _ = coords.shape
        assert num_targets_base == self._num_targets_base

        r = self._radius
        out_pyramid = []
        for idx, corr_data in enumerate(self._corr_pyramid):
            target_indices = corr_data.target_indices
            assert target_indices.ndim == 1
            coords_selected = th.index_select(coords, dim=0, index=target_indices)
            num_targets = target_indices.numel()

            dx = th.linspace(-r, r, 2*r+1, device=coords.device)
            dy = th.linspace(-r, r, 2*r+1, device=coords.device)
            # delta: 2*r+1, 2*r+1, 2
            # NOTE: Unlike in the original implementation, we change the order
            # such that delta[..., 0] corresponds to x and delta[..., 1] corresponds to y
            # In fact, it does not matter since the same targets are looked up and then flattened and fed as channels.
            delta = th.stack(th.meshgrid(dy, dx)[::-1], dim=-1)

            centroid_lvl = coords_selected.reshape(num_targets*batch*h1*w1, 1, 1, 2) / 2**idx
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
            # coords_lvl: num_targets*batch*h1*w1, 2*r+1, 2*r+1, 2
            coords_lvl = centroid_lvl + delta_lvl

            # (reminder) corr_batched: num_targets*batch*h1*w1, 1, h2, w2
            # corr_feat: num_targets*batch*h1*w1, 1, 2*r+1, 2*r+1
            corr_feat = bilinear_sampler(corr_data.corr_batched, coords_lvl)
            # corr_feat: num_targets, batch, h1, w1, (2*r+1)**2
            corr_feat = corr_feat.view(num_targets, batch, h1, w1, -1)
            out_pyramid.append(corr_feat)

        # out: (num_targets_at_lvl_0 + ... + num_targets_at_lvl_final), batch, h1, w1, (2*r+1)**2
        out = th.cat(out_pyramid, dim=0)
        # out: batch, (num_targets_at_lvl_0 + ... + num_targets_at_lvl_final), (2*r+1)**2, h1, w1
        out = out.permute(1, 0, 4, 2, 3)
        # out: batch, (num_targets_at_lvl_0 + ... + num_targets_at_lvl_final)*(2*r+1)**2, h1, w1
        out = out.reshape(batch, -1, h1, w1).float()
        return out