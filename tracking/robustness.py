
from dataclasses import dataclass
from collections.abc import Iterable
from typing import List
import torch

from base.metriclogger import MetricLogger
from tracking.robust_mask import RobustMask
from tracking.correct_mask import CorrectMask
from utils import save_object

@dataclass
class Robustness(MetricLogger):

    dataset_size: int = 0
    batch_size: int = 0
    learned_thres: int = 3 #learned threshold in epochs
    granularity: str = 'by_ep'
    output_location: str = '/'

    def __post_init__(self):

        if self.dataset_size == 0 or self.batch_size == 0:
            raise ValueError('Invalid dataset size...')
            sys.exit(0)

        self.model_outputs = {}
        self.correct_examples = {}
        self.example_order = {}

        self.correct_mask = CorrectMask(dataset_size = dataset_size)

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
        if not ordering:
            raise ValueError('Specify the order in which the examples appear in this base relative to original dataset!')
            sys.exit(1)

        self.model_outputs[self._epoch, self._iteration] = model_outputs
        self.example_order[self._epoch, self._iteration] = ordering

        self.classification = torch.zeros(len(model_outputs))
            for idx, output in enumerate(model_outputs):
                if torch.argmax(output) == self._targets[idx]:
                    self.classification[idx] = 1

        self.correct_mask.set_mask_on(positions = self.classification, ordering = ordering)
        self.correct_examples[self._epoch, self._iteration] = self.correct_mask.mask



    def post_iteration(self) -> None:
        """Functions to execute during once batch is loaded and after optimizer step."""
        self._iteration += 1

    def end_epoch(self) -> None:
        """Functions to execute at the end of an epoch."""
        self._epoch += 1

    def end_training(self) -> None:
        #compute robust mask and write to file
        save_object(object = self.correct_examples,
                    output_location = self.output_location,
                    object_name = 'CorrectExamples')

        save_object(object = self.example_order,
                    output_location = self.output_location,
                    object_name = 'ExampleOrder')


