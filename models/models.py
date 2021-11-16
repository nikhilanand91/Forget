import abc

class Model(abc.ABC):

    @abc.abstractmethod
    def get_model():
        """ Function to get model. """
        pass

    @abc.abstractmethod
    def get_optimizer():
        """ Get the optimizer. """
        pass