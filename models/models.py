import abc

class Model(abc.ABC):

    @abc.abstractmethod
    def forward(self):
        pass

    @abc.abstractmethod
    def get_model(self):
        pass