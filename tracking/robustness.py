from dataclasses import dataclass
from collections.abc import Iterable
from typing import List
import torch

from base.metriclogger import MetricLogger
from tracking.accuracy import Accuracy
import tracking.robust_mask
import utils.save


@dataclass
class Robustness(MetricLogger):

    output_location: str = '/'

    def __post_init__(self):

        self.correct_mask = {}

        self._iteration = 0
        self._epoch = 0

    def description(self) -> str:
        return 'Metric to log robustness statistics.'

    @property
    def epoch(self):
        return self._epoch

    @epoch.setter
    def epoch(self, value):
        if value < 0:
            raise ValueError('Cannot set epoch < 0.')
        self._epoch = value

    @property
    def iteration(self):
        return self._iteration

    @iteration.setter
    def iteration(self, value):
        if value < 0:
            raise ValueError('Cannot set batch counter < 0.')
        self._iteration = value

    def pre_training(self) -> None:
        """Functions to execute before training loop starts."""
        pass

    def start_epoch(self) -> None:
        """Functions to execute at the start of each epoch but before we load a batch."""
        pass

    def pre_iteration(self) -> None:
        """Functions to execute during once batch is loaded but before optimizer step."""
        pass


    def pre_iteration(self, accuracy_metric: Accuracy):
        if not model or not dataloader:
            raise ValueError('Need to specify model and dataloader to measure correctness.')
            sys.exit(1)

        model.eval()
        for batch in dataloader:
            x, y = batch




    def post_iteration(self) -> None:
        """Functions to execute during once batch is loaded and after optimizer step."""
        self._iteration += 1

    def end_epoch(self) -> None:
        """Functions to execute at the end of an epoch."""
        self._epoch += 1

    def end_training(self) -> None:
        #save the mask of correct examples and which examples they are to a file
        