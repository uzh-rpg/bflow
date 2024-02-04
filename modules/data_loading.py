import os
from typing import Dict, Any, Union

import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from data.dsec.provider import DatasetProvider as DatasetProviderDSEC
from data.multiflow2d.provider import DatasetProvider as DatasetProviderMULTIFLOW2D


class DataModule(pl.LightningDataModule):
    DSEC_STR = 'dsec'
    MULTIFLOW2D_REGEN_STR = 'multiflow_regen'

    def __init__(self,
                 config: Union[Dict[str, Any], DictConfig],
                 batch_size_train: int = 1,
                 batch_size_val: int = 1):
        super().__init__()
        dataset_params = config['dataset']
        dataset_type = dataset_params['name']
        num_workers = config['hardware']['num_workers']

        assert dataset_type in {self.DSEC_STR, self.MULTIFLOW2D_REGEN_STR}
        self.dataset_type = dataset_type

        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val

        assert self.batch_size_train >= 1
        assert self.batch_size_val >= 1

        if num_workers is None:
            num_workers = 2*max([batch_size_train, batch_size_val])
            num_workers = min([num_workers, os.cpu_count()])
        print(f'num_workers: {num_workers}')

        self.num_workers = num_workers
        assert self.num_workers >= 0

        nbins_context = config['model']['num_bins']['context']

        if dataset_type == self.DSEC_STR:
            dataset_provider = DatasetProviderDSEC(dataset_params, nbins_context)
        elif dataset_type == self.MULTIFLOW2D_REGEN_STR:
            dataset_provider = DatasetProviderMULTIFLOW2D(dataset_params, nbins_context)
        else:
            raise NotImplementedError
        self.train_dataset = dataset_provider.get_train_dataset()
        if dataset_type == self.DSEC_STR:
            self.val_dataset = None
            self.test_dataset = dataset_provider.get_test_dataset()
        else:
            self.val_dataset = dataset_provider.get_val_dataset()
            self.test_dataset = None

        self.nbins_context = dataset_provider.get_nbins_context()
        self.nbins_correlation = dataset_provider.get_nbins_correlation()

        assert self.nbins_context == nbins_context
        # Fill in nbins_correlation here because it can depend on the dataset.
        if 'correlation' in config['model']['num_bins']:
            nbins_correlation = config['model']['num_bins']['correlation']
            if nbins_correlation is None:
                config['model']['num_bins']['correlation'] = self.nbins_correlation
            else:
                assert nbins_correlation == self.nbins_correlation

    def train_dataloader(self):
        return DataLoader(
                dataset=self.train_dataset,
                batch_size=self.batch_size_train,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True)

    def val_dataloader(self):
        assert self.val_dataset is not None, f'No validation data found for {self.dataset_type} dataset'
        return DataLoader(
                dataset=self.val_dataset,
                batch_size=self.batch_size_val,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True)

    def test_dataloader(self):
        assert self.test_dataset is not None, f'No test data found for {self.dataset_type} dataset'
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False)

    def get_nbins_context(self):
        return self.nbins_context

    def get_nbins_correlation(self):
        return self.nbins_correlation
