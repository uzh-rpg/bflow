from pathlib import Path
from typing import Union

import cv2
import h5py
import numpy as np


def _flow_16bit_to_float(flow_16bit: np.ndarray):
    assert flow_16bit.dtype == np.uint16, flow_16bit.dtype
    assert flow_16bit.ndim == 3
    h, w, c = flow_16bit.shape
    assert c == 3

    valid2D = flow_16bit[..., 2] == 1
    assert valid2D.shape == (h, w)
    assert np.all(flow_16bit[~valid2D, -1] == 0)
    valid_map = np.where(valid2D)
    flow_16bit = flow_16bit.astype('float32')
    flow_map = np.zeros((h, w, 2), dtype='float32')
    flow_map[valid_map[0], valid_map[1], 0] = (flow_16bit[valid_map[0], valid_map[1], 0] - 2 ** 15) / 128
    flow_map[valid_map[0], valid_map[1], 1] = (flow_16bit[valid_map[0], valid_map[1], 1] - 2 ** 15) / 128
    return flow_map, valid2D


def load_flow(flowfile: Path):
    assert flowfile.exists()
    assert flowfile.suffix == '.png'
    # flow_16bit = np.array(Image.open(str(flowfile)))
    flow_16bit = cv2.imread(str(flowfile), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    flow, valid2D = _flow_16bit_to_float(flow_16bit)
    return flow, valid2D


def _blosc_opts(complevel=1, complib='blosc:zstd', shuffle='byte'):
    shuffle = 2 if shuffle == 'bit' else 1 if shuffle == 'byte' else 0
    compressors = ['blosclz', 'lz4', 'lz4hc', 'snappy', 'zlib', 'zstd']
    complib = ['blosc:' + c for c in compressors].index(complib)
    args = {
        'compression': 32001,
        'compression_opts': (0, 0, 0, 0, complevel, shuffle, complib),
    }
    if shuffle > 0:
        # Do not use h5py shuffle if blosc shuffle is enabled.
        args['shuffle'] = False
    return args


def np_array_to_h5(array: np.ndarray, outpath: Path) -> None:
    isinstance(array, np.ndarray)
    assert outpath.suffix == '.h5'

    with h5py.File(str(outpath), 'w') as h5f:
        h5f.create_dataset('voxel_grid', data=array, shape=array.shape, dtype=array.dtype,
                           **_blosc_opts(complevel=1, shuffle='byte'))


def h5_to_np_array(inpath: Path) -> Union[np.ndarray, None]:
    assert inpath.suffix == '.h5'
    assert inpath.exists()

    try:
        with h5py.File(str(inpath), 'r') as h5f:
            array = np.asarray(h5f['voxel_grid'])
            return array
    except OSError as e:
        print(f'Error loading {inpath}')
    return None
