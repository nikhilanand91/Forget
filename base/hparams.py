import abc

class HParams(abc.ABC):
    """An instance of hyperparameters. This could be, e.g.,
    training hyperparameters or those used for robustness
    experiments."""

    """
    @abc.abstractmethod
    def create_from_hparams(self) -> 'HParams':
        #Create the object from specified hparams, which
        #can then be passed onto a training loop.
        pass
    """