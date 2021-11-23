from tracking.metriclogger import MetricLogger
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

        self.learned_masks = list()
        self.correct_examples = list()
        self.robust_mask = RobustMask()

	
	def description(self) -> str:
    	return 'Metric to log robustness statistics.'

    def pre_training(self) -> None:
        """Functions to execute before training loop starts."""
        pass

    def start_epoch(self) -> None:
        """Functions to execute at the start of each epoch but before we load a batch."""
        order = ...

    def pre_iteration(self, model_outputs: torch.Tensor, targets: torch.Tensor, batch_id: int)
        """Functions to execute during once batch is loaded but before optimizer step."""
        classification = classify_batch(model_outputs, targets)



    def post_iteration(self) -> None:
        """Functions to execute during once batch is loaded and after optimizer step."""

    def end_epoch(self) -> None:
        """Functions to execute at the end of an epoch."""



