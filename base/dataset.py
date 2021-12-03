import abc

class Dataset(abc.ABC):

    @abc.abstractmethod
    def get_dataset(self):
        """Get the dataset."""
        pass

    @abc.abstractmethod
    def get_dataloader(self):
        """Get the dataloader."""
        pass

    @abc.abstractmethod
    def set_transforms(self):
        """Whatever transforms we need to apply to dataset."""
        pass

    @abc.abstractmethod
    def get_sampler(self):
        """Get the sampler used e.g. random, sequential, etc."""

    @abc.abstractmethod
    def get_order(self):
        """Get the ordering of examples relative to original dataset."""