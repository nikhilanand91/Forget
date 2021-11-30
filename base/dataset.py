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
    def get_relative_ordering(self):
        """Get relative ordering of examples in a batch to a dataloader.
        Useful utility function when dataloader is shuffled between epochs."""
        pass

    @abc.abstractmethod
    def set_transforms(self):
        """Whatever transforms we need to apply to dataset."""
        pass