from pathlib import Path
from typing import Dict

import h5py
import imageio as iio
import numpy as np
import torch
from torch.nn.functional import interpolate

from data.utils.generic import np_array_to_h5, h5_to_np_array
from data.utils.representations import VoxelGrid


class Sample:
    # Assumes the following structure:
    # seq*
    # ├── events
    # │   └── events.h5
    # ├── flow
    # │   ├── 0500000.h5
    # │   ├── ...
    # │   └── 0900000.h5
    # └── images
    #     ├── 0400000.png
    #     ├── ...
    #     └── 0900000.png

    def __init__(self,
                 sample_path: Path,
                 height: int,
                 width: int,
                 num_bins_context: int,
                 load_voxel_grid: bool=True,
                 extended_voxel_grid: bool=True,
                 downsample: bool=False,
                 ) -> None:
        assert sample_path.is_dir()
        assert num_bins_context >= 1
        assert sample_path.is_dir()

        nbins_context2corr = {
            6: 4,
            11: 7,
            21: 13,
            41: 25,
        }
        nbins_context2deltatime = {
            6: 100000,
            11: 50000,
            21: 25000,
            41: 12500,
        }

        ### To downsample or to not downsample
        self.downsample = downsample

        ### Voxel Grid
        assert num_bins_context in nbins_context2corr.keys()
        self.num_bins_context = num_bins_context
        self.num_bins_correlation = nbins_context2corr[num_bins_context]
        # We subtract one because the bin at the reference time is redundant
        self.num_bins_total = self.num_bins_context + self.num_bins_correlation - 1

        self.voxel_grid = VoxelGrid(self.num_bins_total, height, width)

        # Image data
        ref_time_us = 400*1000
        target_time_us = 900*1000
        img_ref_path = sample_path / 'images' / (f'{ref_time_us}'.zfill(7) + '.png')
        assert img_ref_path.exists()
        img_target_path = sample_path / 'images' / (f'{target_time_us}'.zfill(7) + '.png')
        assert img_target_path.exists()
        self.img_filepaths = [img_ref_path, img_target_path]
        self.img_ts = [int(x.stem) for x in self.img_filepaths]

        # Extract timestamps for later retrieving event data
        self.bin_0_time = self.img_ts[0] - (self.num_bins_correlation - 1)*nbins_context2deltatime[num_bins_context]
        assert self.bin_0_time >= 0
        self.bin_target_time = self.img_ts[1]

        # Flow data
        self.flow_ref_ts_us = ref_time_us
        flow_dir = sample_path / 'flow'
        assert flow_dir.is_dir()
        flow_filepaths = list()
        for flow_file in flow_dir.iterdir():
            assert flow_file.suffix == '.h5'
            flow_filepaths.append(flow_file)
        flow_filepaths.sort()
        self.flow_filepaths = flow_filepaths
        self.flow_ts_us = [int(x.stem) for x in self.flow_filepaths]

        # Event data
        ev_dir = sample_path / 'events'
        assert ev_dir.is_dir()
        self.event_filepath = ev_dir / 'events.h5'
        assert self.event_filepath.exists()

        ### Voxel Grid Saving
        self.version = 1 if extended_voxel_grid else 0
        downsample_str = '_downsampled' if self.downsample else ''
        self.voxel_grid_file = ev_dir / f'voxel_grid_v{self.version}_{self.num_bins_total}_bins{downsample_str}.h5'
        self.load_voxel_grid_from_disk = load_voxel_grid

    def downsample_tensor(self, input_tensor: torch.Tensor):
        assert input_tensor.ndim == 3
        assert self.downsample
        ch, ht, wd = input_tensor.shape
        input_tensor = input_tensor.float()
        return interpolate(input_tensor[None, ...], size=(ht//2, wd//2), align_corners=True, mode='bilinear').squeeze()

    def get_flow_gt(self, flow_every_n_ms: int):
        assert flow_every_n_ms > 0
        assert flow_every_n_ms % 10 == 0, 'must be a multiple of 10'
        delta_ts_us = flow_every_n_ms*1000
        out = {
            'flow': list(),
            'timestamps': list(),
        }
        for flow_ts_us, flow_filepath in zip(self.flow_ts_us, self.flow_filepaths):
            if (flow_ts_us - self.flow_ref_ts_us) % delta_ts_us != 0:
                continue
            out['timestamps'].append(flow_ts_us)
            with h5py.File(str(flow_filepath), 'r') as h5f:
                flow = np.asarray(h5f['flow'])
                # h, w, c -> c, h, w
                flow = np.moveaxis(flow, -1, 0)
                flow = torch.from_numpy(flow)
                if self.downsample:
                    flow = self.downsample_tensor(flow)
                    flow = flow/2
                out['flow'].append(flow)
        return out

    def get_images(self):
        out = {
            'images': list(),
            'timestamps': self.img_ts,
        }
        for img_path in self.img_filepaths:
            img = np.asarray(iio.imread(str(img_path), format='PNG-FI'))
            # img: (h, w, c) -> (c, h, w)
            img = np.moveaxis(img, -1, 0)
            img = torch.from_numpy(img)
            if self.downsample:
                img = self.downsample_tensor(img)
            out['images'].append(img)
        return out

    def _get_events(self, t_start: int, t_end: int):
        assert t_start >= 0
        assert t_end <= 1000000
        assert t_end > t_start
        with h5py.File(str(self.event_filepath), 'r') as h5f:
            time = np.asarray(h5f['t'])
            first_idx = np.searchsorted(time, t_start, side='left')
            last_idx_p1 = np.searchsorted(time, t_end, side='right')
            out = {
                'x': np.asarray(h5f['x'][first_idx:last_idx_p1]),
                'y': np.asarray(h5f['y'][first_idx:last_idx_p1]),
                'p': np.asarray(h5f['p'][first_idx:last_idx_p1]),
                't': time[first_idx:last_idx_p1],
            }
        return out

    def _events_to_voxel_grid(self, event_dict: Dict[str, np.ndarray], t0_center: int=None, t1_center: int=None) -> torch.Tensor:
        # int32 is enough for this dataset as the timestamps are in us and start at 0 for every sample sequence
        t = event_dict['t'].astype('int32')
        x = event_dict['x'].astype('int16')
        y = event_dict['y'].astype('int16')
        pol = event_dict['p'].astype('int8')
        return self.voxel_grid.convert(
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(pol),
            torch.from_numpy(t),
            t0_center,
            t1_center)

    def _construct_voxel_grid(self, ts_from: int, ts_to: int):
        if self.version == 1:
            t_start, t_end = self.voxel_grid.get_extended_time_window(ts_from, ts_to)
            assert (ts_from - t_start) <= 100000, f'ts_from: {ts_from}, t_start: {t_start}'
            assert (t_end - ts_to) <= 100000, f't_end: {t_end}, ts_to: {ts_to}'
            event_data = self._get_events(t_start, t_end)
            voxel_grid = self._events_to_voxel_grid(event_data, ts_from, ts_to)
        elif self.version == 0:
            event_data = self._get_events(ts_from, ts_to)
            voxel_grid = self._events_to_voxel_grid(event_data)
        else:
            raise NotImplementedError
        if self.downsample:
            voxel_grid = self.downsample_tensor(voxel_grid)
        return voxel_grid

    def _load_or_save_voxel_grid(self, ts_from: int, ts_to: int) -> torch.Tensor:
        if self.voxel_grid_file.exists():
            voxel_grid_numpy = h5_to_np_array(self.voxel_grid_file)
            if voxel_grid_numpy is not None:
                # Squeeze because we may have saved it with batch dim 1 before (by mistake)
                return torch.from_numpy(voxel_grid_numpy).squeeze()
            # If None is returned, it means that the file was corrupt and we overwrite it.
        voxel_grid = self._construct_voxel_grid(ts_from, ts_to)
        np_array_to_h5(voxel_grid.numpy(), self.voxel_grid_file)
        return voxel_grid

    def get_voxel_grid(self) -> torch.Tensor:
        ts_from = self.bin_0_time
        ts_to = self.bin_target_time
        if self.load_voxel_grid_from_disk:
            return self._load_or_save_voxel_grid(ts_from, ts_to).squeeze()
        return self._construct_voxel_grid(ts_from, ts_to).squeeze()

    def voxel_grid_bin_idx_for_reference(self) -> int:
        return self.num_bins_correlation - 1