import os
from pathlib import Path
from typing import List, Optional
import weakref

import h5py
import imageio as iio
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from data.dsec.eventslicer import EventSlicer
from data.utils.augmentor import FlowAugmentor
from data.utils.generic import np_array_to_h5, h5_to_np_array
from data.utils.representations import VoxelGrid, norm_voxel_grid


class BaseSubSequence(Dataset):
    # seq_name (e.g. zurich_city_10_a)
    # ├── flow
    # │   ├── backward (for train only)
    # │   │   ├── xxxxxx.png
    # │   │   └── ...
    # │   ├── backward_timestamps.txt (for train only)
    # │   ├── forward (for train only)
    # │   │   ├── xxxxxx.png
    # │   │   └── ...
    # │   └── forward_timestamps.txt (for train and test)
    # └── events
    #     ├── left
    #     │   ├── events.h5
    #     │   └── rectify_map.h5
    #     └── right
    #         ├── events.h5
    #         └── rectify_map.h5
    #
    # For now this class
    # - only returns forward flow and the corresponding event representation (of the left event camera)
    # - does not implement recurrent loading of data
    # - only returns a forward rectified voxel grid with a fixed number of bins
    # - is not implemented to offer multi-dataset training schedules (like in RAFT)
    def __init__(self,
            seq_path: Path,
            forward_flow_timestamps: np.ndarray,
            forward_flow_paths: List[Path],
            data_augm: bool,
            num_bins: int=15,
            load_voxel_grid: bool=True,
            extended_voxel_grid: bool=True,
            normalize_voxel_grid: bool=False):
        assert num_bins >= 1
        assert seq_path.is_dir()

        # Save output dimensions
        self.height = 480
        self.width = 640
        self.num_bins = num_bins

        self.augmentor = FlowAugmentor(crop_size_hw=(288, 384)) if data_augm else None

        # Set event representation
        self.voxel_grid = VoxelGrid(self.num_bins, self.height, self.width)
        self.normalize_voxel_grid: Optional[norm_voxel_grid] = norm_voxel_grid if normalize_voxel_grid else None

        assert len(forward_flow_paths) == forward_flow_timestamps.shape[0]

        self.forward_flow_timestamps = forward_flow_timestamps

        for entry in forward_flow_paths:
            assert entry.exists()
            assert str(entry.name).endswith('.png')
        self.forward_flow_list = forward_flow_paths

        ### prepare loading of event data and load rectification map
        self.ev_dir = seq_path / 'events' / 'left'
        assert self.ev_dir.is_dir()
        self.ev_file = self.ev_dir / 'events.h5'
        assert self.ev_file.exists()

        rectify_events_map_file = self.ev_dir / 'rectify_map.h5'
        assert rectify_events_map_file.exists()
        with h5py.File(str(rectify_events_map_file), 'r') as h5_rect:
            self.rectify_events_map = h5_rect['rectify_map'][()]

        self.h5f: Optional[h5py.File] = None
        self.event_slicer: Optional[EventSlicer] = None
        self.h5f_opened = False

        ### prepare loading of image data (in left event camera frame)
        img_dir_ev_left = seq_path / 'images' / 'left' / 'ev_inf'
        self.img_dir_ev_left = None if not img_dir_ev_left.is_dir() else img_dir_ev_left

        ### Voxel Grid Saving
        # Version 0:    Without considering the boundary properly but strictly causal
        # Version 1:    Considering the boundary effect but loads a few events "of the future"
        self.version = 1 if extended_voxel_grid else 0
        self.voxel_grid_dir = self.ev_dir / f'voxel_grids_v{self.version}_100ms_forward_{num_bins}_bins'
        self.load_voxel_grid = load_voxel_grid
        if self.load_voxel_grid:
            if not self.voxel_grid_dir.exists():
                os.mkdir(self.voxel_grid_dir)
            else:
                assert self.voxel_grid_dir.is_dir()

    def __open_h5f(self):
        assert self.h5f is None
        assert self.event_slicer is None

        self.h5f = h5py.File(str(self.ev_file), 'r')
        self.event_slicer = EventSlicer(self.h5f)

        self._finalizer = weakref.finalize(self, self.__close_callback, self.h5f)
        self.h5f_opened = True

    def _events_to_voxel_grid(self, x, y, p, t, t0_center: int=None, t1_center: int=None):
        t = t.astype('int64')
        x = x.astype('float32')
        y = y.astype('float32')
        pol = p.astype('float32')
        return self.voxel_grid.convert(
                torch.from_numpy(x),
                torch.from_numpy(y),
                torch.from_numpy(pol),
                torch.from_numpy(t),
                t0_center,
                t1_center)

    def getHeightAndWidth(self):
        return self.height, self.width

    @staticmethod
    def __close_callback(h5f: h5py.File):
        assert h5f is not None
        h5f.close()

    def _rectify_events(self, x: np.ndarray, y: np.ndarray):
        # From distorted to undistorted
        rectify_map = self.rectify_events_map
        assert rectify_map.shape == (self.height, self.width, 2), rectify_map.shape
        assert x.max() < self.width
        assert y.max() < self.height
        return rectify_map[y, x]

    def _get_ev_left_img(self, img_file_idx: int):
        if self.img_dir_ev_left is None:
            return None
        img_filename = f'{img_file_idx}'.zfill(6) + '.png'
        img_filepath = self.img_dir_ev_left / img_filename
        if not img_filepath.exists():
            return None
        img = np.asarray(iio.imread(str(img_filepath), format='PNG-FI'))
        # img: (h, w, c) -> (c, h, w)
        img = np.moveaxis(img, -1, 0)
        return img

    def _get_events(self, ts_from: int, ts_to: int, rectify: bool):
        if not self.h5f_opened:
            self.__open_h5f()

        start_time_us = self.event_slicer.get_start_time_us()
        final_time_us = self.event_slicer.get_final_time_us()
        assert ts_from > start_time_us - 50000, 'Do not request more than 50 ms before the minimum time. Otherwise, something might be wrong.'
        assert ts_to < final_time_us + 50000, 'Do not request more than 50 ms past the maximum time. Otherwise, something might be wrong.'
        if ts_from < start_time_us:
            # Take the minimum time instead to avoid assertions in the event slicer.
            ts_from = start_time_us
        if ts_to > final_time_us:
            # Take the maximum time instead to avoid assertions in the event slicer.
            ts_to = final_time_us
        assert ts_from < ts_to

        event_data = self.event_slicer.get_events(ts_from, ts_to)

        pol = event_data['p']
        time = event_data['t']
        x = event_data['x']
        y = event_data['y']

        if rectify:
            xy_rect = self._rectify_events(x, y)
            x = xy_rect[:, 0]
            y = xy_rect[:, 1]
        assert pol.shape == time.shape == x.shape == y.shape

        out = {
            'pol': pol,
            'time': time,
            'x': x,
            'y': y,
        }
        return out

    def _construct_voxel_grid(self, ts_from: int, ts_to: int, rectify: bool=True):
        if self.version == 1:
            t_start, t_end = self.voxel_grid.get_extended_time_window(ts_from, ts_to)
            assert (ts_from - t_start) < 50000, f'ts_from: {ts_from}, t_start: {t_start}'
            assert (t_end - ts_to) < 50000, f't_end: {t_end}, ts_to: {ts_to}'
            event_data = self._get_events(t_start, t_end, rectify=rectify)
            voxel_grid = self._events_to_voxel_grid(event_data['x'], event_data['y'], event_data['pol'], event_data['time'], ts_from, ts_to)
            return voxel_grid
        elif self.version == 0:
            event_data = self._get_events(ts_from, ts_to, rectify=rectify)
            voxel_grid = self._events_to_voxel_grid(event_data['x'], event_data['y'], event_data['pol'], event_data['time'])
            return voxel_grid
        raise NotImplementedError

    def _load_voxel_grid(self, ts_from: int, ts_to: int, file_index: int):
        assert file_index >= 0

        # Assuming we want to load the 'forward' voxel grid.
        voxel_grid_file = self.voxel_grid_dir / (f'{file_index}'.zfill(6) + '.h5')
        if not voxel_grid_file.exists():
            voxel_grid = self._construct_voxel_grid(ts_from, ts_to)
            np_array_to_h5(voxel_grid.numpy(), voxel_grid_file)
            return voxel_grid
        return torch.from_numpy(h5_to_np_array(voxel_grid_file))

    def _get_voxel_grid(self, ts_from: int, ts_to: int, file_index: int):
        if self.load_voxel_grid:
            return self._load_voxel_grid(ts_from, ts_to, file_index)
        return self._construct_voxel_grid(ts_from, ts_to)