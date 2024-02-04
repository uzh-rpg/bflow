import math
from typing import Optional

import torch
torch.set_num_threads(1) # intraop parallelism (this can be a good option)
torch.set_num_interop_threads(1) # interop parallelism


def norm_voxel_grid(voxel_grid: torch.Tensor):
    mask = torch.nonzero(voxel_grid, as_tuple=True)
    if mask[0].size()[0] > 0:
        mean = voxel_grid[mask].mean()
        std = voxel_grid[mask].std()
        if std > 0:
            voxel_grid[mask] = (voxel_grid[mask] - mean) / std
        else:
            voxel_grid[mask] = voxel_grid[mask] - mean
    return voxel_grid


class EventRepresentation:
    def convert(self, x: torch.Tensor, y: torch.Tensor, pol: torch.Tensor, time: torch.Tensor, t_from: Optional[int]=None, t_to: Optional[int]=None):
        raise NotImplementedError


class VoxelGrid(EventRepresentation):
    def __init__(self, channels: int, height: int, width: int):
        assert channels > 1
        assert height > 1
        assert width > 1
        self.nb_channels = channels
        self.height = height
        self.width = width

    def get_extended_time_window(self, t0_center: int, t1_center: int):
        dt = self._get_dt(t0_center, t1_center)
        t_start = math.floor(t0_center - dt)
        t_end = math.ceil(t1_center + dt)
        return t_start, t_end

    def _construct_empty_voxel_grid(self):
        return torch.zeros(
            (self.nb_channels, self.height, self.width),
            dtype=torch.float,
            requires_grad=False,
            device=torch.device('cpu'))

    def _get_dt(self, t0_center: int, t1_center: int):
        assert t1_center > t0_center
        return (t1_center - t0_center)/(self.nb_channels - 1)

    def _normalize_time(self, time: torch.Tensor, t0_center: int, t1_center: int):
        # time_norm < t0_center will be negative
        # time_norm == t0_center is 0
        # time_norm > t0_center is positive
        # time_norm == t1_center is (nb_channels - 1)
        # time_norm > t1_center is greater than (nb_channels - 1)
        return (time - t0_center)/(t1_center - t0_center)*(self.nb_channels - 1)

    @staticmethod
    def _is_int_tensor(tensor: torch.Tensor) -> bool:
        return not torch.is_floating_point(tensor) and not torch.is_complex(tensor)

    def convert(self, x: torch.Tensor, y: torch.Tensor, pol: torch.Tensor, time: torch.Tensor, t0_center: Optional[int]=None, t1_center: Optional[int]=None):
        assert x.device == y.device == pol.device == time.device == torch.device('cpu')
        assert type(t0_center) == type(t1_center)
        assert x.shape == y.shape == pol.shape == time.shape
        assert x.ndim == 1
        assert self._is_int_tensor(time)

        is_int_xy = self._is_int_tensor(x)
        if is_int_xy:
            assert self._is_int_tensor(y)

        voxel_grid = self._construct_empty_voxel_grid()
        ch, ht, wd = self.nb_channels, self.height, self.width
        with torch.no_grad():
            t0_center = t0_center if t0_center is not None else time[0]
            t1_center = t1_center if t1_center is not None else time[-1]
            t_norm = self._normalize_time(time, t0_center, t1_center)

            t0 = t_norm.floor().int()
            value = 2*pol.float()-1

            if is_int_xy:
                for tlim in [t0,t0+1]:
                    mask = (tlim >= 0) & (tlim < ch)
                    interp_weights = value * (1 - (tlim - t_norm).abs())

                    index = ht * wd * tlim.long() + \
                            wd * y.long() + \
                            x.long()

                    voxel_grid.put_(index[mask], interp_weights[mask], accumulate=True)
            else:
                x0 = x.floor().int()
                y0 = y.floor().int()
                for xlim in [x0,x0+1]:
                    for ylim in [y0,y0+1]:
                        for tlim in [t0,t0+1]:

                            mask = (xlim < wd) & (xlim >= 0) & (ylim < ht) & (ylim >= 0) & (tlim >= 0) & (tlim < ch)
                            interp_weights = value * (1 - (xlim-x).abs()) * (1 - (ylim-y).abs()) * (1 - (tlim - t_norm).abs())

                            index = ht * wd * tlim.long() + \
                                    wd * ylim.long() + \
                                    xlim.long()

                            voxel_grid.put_(index[mask], interp_weights[mask], accumulate=True)

        return voxel_grid
