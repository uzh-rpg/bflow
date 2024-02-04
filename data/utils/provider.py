import abc


class DatasetProviderBase(abc.ABC):
    @abc.abstractmethod
    def get_train_dataset(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_val_dataset(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_test_dataset(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_nbins_context(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_nbins_correlation(self):
        raise NotImplementedError