from base.metriclogger import MetricLogger
from tracking.classification_utils import classify_batch
from dataclasses import dataclass
from tracking.robust_mask import RobustMask
import torch
from typing import List

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

        self.learned_examples = list()
        self.correct_examples = list()

        self._model_outputs = None
        self._targets = None
        self._batch_counter = 0
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
    def batch_counter(self):
        return self._batch_counter

    @batch_counter.setter
    def batch_counter(self, value):
        if value < 0:
            raise ValueError('Cannot set batch counter < 0.')
        self._batch_counter = value

    def pre_training(self) -> None:
        """Functions to execute before training loop starts."""
        pass

    def start_epoch(self) -> None:
        """Functions to execute at the start of each epoch but before we load a batch."""
        pass

    def pre_iteration(self, model_outputs: torch.Tensor, targets: torch.Tensor) -> None:
        """Functions to execute during once batch is loaded but before optimizer step."""
        self._model_outputs = model_outputs
        self._targets = targets
        self.classification = torch.zeros(len(self._model_outputs))
            for idx, output in enumerate(self._model_outputs):
                if torch.argmax(output) == self._targets[idx]:
                    self.classification[idx] = 1

        self.robust_mask.

    def post_iteration(self) -> None:
        """Functions to execute during once batch is loaded and after optimizer step."""

    def end_epoch(self) -> None:
        """Functions to execute at the end of an epoch."""



