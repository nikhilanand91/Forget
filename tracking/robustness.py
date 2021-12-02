
from dataclasses import dataclass
from collections.abc import Iterable
from typing import List
import torch

from base.metriclogger import MetricLogger
from tracking.robust_mask import RobustMask
from tracking.correct_mask import CorrectMask

@dataclass
class Robustness(MetricLogger):

    dataset_size: int = 0
    batch_size: int = 0
    learned_thres: int = 3 #learned threshold in epochs
    granularity: str = 'by_ep'

    def __post_init__(self):

        if self.dataset_size == 0 or self.batch_size == 0:
            raise ValueError('Invalid dataset size...')
            sys.exit(0)

        self.model_outputs = {}
        self.correct_examples = {}
        self.correct_mask = CorrectMask(dataset_size = dataset_size)

        self._model_outputs = None
        self._targets = None
        self._iteration = 0
        self._epoch = 0
        self.classification = None

	
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

    @batch_counter.setter
    def batch_counter(self, value):
        if value < 0:
            raise ValueError('Cannot set batch counter < 0.')
        self._iteration = value

    def pre_training(self) -> None:
        """Functions to execute before training loop starts."""
        pass

    def start_epoch(self) -> None:
        """Functions to execute at the start of each epoch but before we load a batch."""
        pass

    def pre_iteration(self, model_outputs: torch.Tensor, targets: torch.Tensor, ordering = None) -> None:
        """Functions to execute during once batch is loaded but before optimizer step."""
        self._model_outputs = model_outputs
        self._targets = targets
        self.classification = torch.zeros(len(self._model_outputs))
            for idx, output in enumerate(self._model_outputs):
                if torch.argmax(output) == self._targets[idx]:
                    self.classification[idx] = 1

        self.correct_mask.set_mask_on(positions = self.classification, ordering = ordering)
        self.correct_examples[self._epoch, self._iteration] = self.correct_mask

        #save ordering, outputs

    def post_iteration(self) -> None:
        """Functions to execute during once batch is loaded and after optimizer step."""
        self._iteration += 1

    def end_epoch(self) -> None:
        """Functions to execute at the end of an epoch."""
        self._epoch += 1

    def end_training(self) -> None:
        #compute robust mask and write to file
        pass


