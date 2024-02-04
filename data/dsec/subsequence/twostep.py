from pathlib import Path
from typing import List

import numpy as np
import torch

from data.dsec.subsequence.base import BaseSubSequence
from data.utils.generic import load_flow
from data.utils.keys import DataLoading, DataSetType


class TwoStepSubSequence(BaseSubSequence):
    def __init__(self,
            seq_path: Path,
            forward_flow_timestamps: np.ndarray,
            forward_flow_paths: List[Path],
            data_augm: bool,
            num_bins: int,
            load_voxel_grid: bool,
            extended_voxel_grid: bool,
            normalize_voxel_grid: bool,
            merge_grids: bool):
        super().__init__(
            seq_path,
            forward_flow_timestamps,
            forward_flow_paths,
            data_augm,
            num_bins,
            load_voxel_grid,
            extended_voxel_grid=extended_voxel_grid,
            normalize_voxel_grid=normalize_voxel_grid)
        self.merge_grids = merge_grids

    def __len__(self):
        return len(self.forward_flow_list)

    def __getitem__(self, index):
        forward_flow_gt_path = self.forward_flow_list[index]
        flow_file_index = int(forward_flow_gt_path.stem)
        forward_flow, forward_flow_valid2D = load_flow(forward_flow_gt_path)
        # forward_flow: (h, w, 2) -> (2, h, w)
        forward_flow = np.moveaxis(forward_flow, -1, 0)

        # Events during the flow duration.
        ev_repr_list = list()
        ts_from = None
        ts_to = None
        for idx in [index, index - 1]:
            if self._is_index_valid(idx):
                ts = self.forward_flow_timestamps[idx]
                ts_from = ts[0]
                ts_to = ts[1]
            else:
                assert idx == index - 1
                assert ts_from is not None
                assert ts_to is not None
                dt = ts_to - ts_from
                ts_to = ts_from
                ts_from = ts_from - dt
            # Hardcoded assumption about 100ms steps and filenames.
            file_index = flow_file_index if idx == index else flow_file_index - 2
            ev_repr = self._get_voxel_grid(ts_from, ts_to, file_index)
            ev_repr_list.append(ev_repr)

        imgs_list = None
        img_idx_reference = flow_file_index
        img_reference = self._get_ev_left_img(img_idx_reference)
        if img_reference is not None:
            # Assume 100ms steps (take every second frame).
            # Assume forward flow (target frame in the future)
            img_idx_target = img_idx_reference + 2
            img_target = self._get_ev_left_img(img_idx_target)
            assert img_target is not None
            imgs_list = [img_reference, img_target]

        # 0: previous, 1: current
        ev_repr_list.reverse()
        if self.merge_grids:
            ev_repr_0 = ev_repr_list[0]
            ev_repr_1 = ev_repr_list[1]
            assert (ev_repr_0[-1] - ev_repr_1[0]).flatten().abs().max() < 0.5, f'{(ev_repr_0[-1] - ev_repr_1[0]).flatten().abs().max()}'
            # Remove the redundant temporal slice.
            event_representations = torch.cat((ev_repr_0, ev_repr_1[1:, ...]), dim=0)
            if self.normalize_voxel_grid is not None:
                event_representations = self.normalize_voxel_grid(event_representations)
        else:
            if self.normalize_voxel_grid is not None:
                ev_repr_list = [self.normalize_voxel_grid(voxel_grid) for voxel_grid in ev_repr_list]
            event_representations = torch.stack(ev_repr_list)

        if self.augmentor is not None:
            event_representations, forward_flow, forward_flow_valid2D, imgs_list = self.augmentor(event_representations, forward_flow, forward_flow_valid2D, imgs_list)

        output = {
            DataLoading.FLOW: forward_flow,
            DataLoading.FLOW_VALID: forward_flow_valid2D,
            DataLoading.FILE_INDEX: flow_file_index,
            # For now returns: 2 x bins x height x width or (2*bins-1) x height x width
            DataLoading.EV_REPR: event_representations,
            DataLoading.DATASET_TYPE: DataSetType.DSEC,
        }
        if imgs_list is not None:
            output.update({DataLoading.IMG: imgs_list})

        return output

    def _is_index_valid(self, index):
        return index >= 0 and index < len(self)
