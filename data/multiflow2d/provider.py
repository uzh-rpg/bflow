import copy
from pathlib import Path
from typing import Dict, Any

import torch.utils.data

from data.utils.provider import DatasetProviderBase
from data.multiflow2d.datasubset import Datasubset


class DatasetProvider(DatasetProviderBase):
    def __init__(self,
                 dataset_params: Dict[str, Any],
                 nbins_context: int):
        dataset_path = Path(dataset_params['path'])
        train_path = dataset_path / 'train'
        val_path = dataset_path / 'val'
        assert dataset_path.is_dir(), str(dataset_path)
        assert train_path.is_dir(), str(train_path)
        assert val_path.is_dir(), str(val_path)

        return_img = True
        return_img_key = 'return_img'
        if return_img_key in dataset_params:
            return_img = dataset_params[return_img_key]
        return_ev = True
        return_ev_key = 'return_ev'
        if return_ev_key in dataset_params:
            return_ev = dataset_params[return_ev_key]
        base_args = {
            'num_bins_context': nbins_context,
            'load_voxel_grid': dataset_params['load_voxel_grid'],
            'normalize_voxel_grid': dataset_params['normalize_voxel_grid'],
            'extended_voxel_grid': dataset_params['extended_voxel_grid'],
            'flow_every_n_ms': dataset_params['flow_every_n_ms'],
            'downsample': dataset_params['downsample'],
            'photo_augm': dataset_params['photo_augm'],
            return_img_key: return_img,
            return_ev_key: return_ev,
        }
        train_args = copy.deepcopy(base_args)
        train_args.update({'data_augm': True})
        val_test_args = copy.deepcopy(base_args)
        val_test_args.update({'data_augm': False})

        train_dataset = Datasubset(train_path, **train_args)
        self.nbins_context = train_dataset.get_num_bins_context()
        self.nbins_correlation = train_dataset.get_num_bins_correlation()

        self.train_dataset = train_dataset
        self.val_dataset = Datasubset(val_path, **val_test_args)
        assert self.val_dataset.get_num_bins_context() == self.nbins_context
        assert self.val_dataset.get_num_bins_correlation() == self.nbins_correlation

    def get_train_dataset(self):
        return self.train_dataset

    def get_val_dataset(self):
        return self.val_dataset

    def get_test_dataset(self):
        raise NotImplementedError

    def get_nbins_context(self):
        return self.nbins_context

    def get_nbins_correlation(self):
        return self.nbins_correlation
