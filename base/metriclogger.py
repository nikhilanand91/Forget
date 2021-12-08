import abc

class MetricLogger(abc.ABC):
    """
    An instance of an metric logger.
    """

    @abc.abstractmethod
    def description(self) -> str:
        """A description of this tracker."""
        pass

    @abc.abstractmethod
    def needs(self) -> dict:
        """What objects this metric logger needs to make measurements."""
        pass

    @abc.abstractmethod
    def pre_training(self) -> None:
        """Functions to execute before training loop starts."""

    @abc.abstractmethod
    def start_epoch(self) -> None:
        """Functions to execute at the start of each epoch but before we load a batch."""

    @abc.abstractmethod
    def pre_iteration(self) -> None:
        """Functions to execute during once batch is loaded but before optimizer step."""

    @abc.abstractmethod
    def post_iteration(self) -> None:
        """Functions to execute during once batch is loaded and after optimizer step."""

    @abc.abstractmethod
    def end_epoch(self) -> None:
        """Functions to execute at the end of an epoch."""
    