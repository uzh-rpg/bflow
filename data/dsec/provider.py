import copy
from pathlib import Path
from typing import Dict, Any

import torch

from data.dsec.sequence import generate_sequence
from data.dsec.subsequence.twostep import TwoStepSubSequence
from data.utils.provider import DatasetProviderBase


class DatasetProvider(DatasetProviderBase):
    def __init__(self,
            dataset_params: Dict[str, Any],
            nbins_context: int):
        dataset_path = Path(dataset_params['path'])

        train_path = dataset_path / 'train'
        test_path = dataset_path / 'test'
        assert dataset_path.is_dir(), str(dataset_path)
        assert train_path.is_dir(), str(train_path)
        assert test_path.is_dir(), str(test_path)

        # NOTE: For now, we assume that the number of bins for the correlation are the same as for the context.
        self.nbins = nbins_context

        subseq_class = TwoStepSubSequence

        base_args = {
            'num_bins': self.nbins,
            'load_voxel_grid': dataset_params['load_voxel_grid'],
            'extended_voxel_grid': dataset_params['extended_voxel_grid'],
            'normalize_voxel_grid': dataset_params['normalize_voxel_grid'],
        }
        base_args.update({'merge_grids': True})
        train_args = copy.deepcopy(base_args)
        train_args.update({'data_augm': True})
        test_args = copy.deepcopy(base_args)
        test_args.update({'data_augm': False})

        train_sequences = list()
        for child in train_path.iterdir():
            sequence = generate_sequence(child, subseq_class, train_args)
            if sequence is not None:
                train_sequences.append(sequence)

        self.train_dataset = torch.utils.data.ConcatDataset(train_sequences)

        # TODO: write specialized test sequence
        #test_sequences = list()
        #for child in test_path.iterdir():
        #    sequence = generate_test_sequence(child, subseq_class, test_args)
        #    if sequence is not None:
        #        test_sequences.append(sequence)
        #self.test_dataset = torch.utils.data.ConcatDataset(test_sequences)
        self.test_dataset = None

    def get_train_dataset(self):
        return self.train_dataset

    def get_val_dataset(self):
        raise NotImplementedError

    def get_test_dataset(self):
        return self.test_dataset

    def get_nbins_context(self):
        return self.nbins

    def get_nbins_correlation(self):
        return self.nbins
