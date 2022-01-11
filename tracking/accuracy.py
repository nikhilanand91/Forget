from dataclasses import dataclass
from collections.abc import Iterable
from typing import List
import torch

from base.metriclogger import MetricLogger
import tracking.difficulty_mask

import utils.save

@dataclass
class Accuracy(MetricLogger):

    output_location: str = '/'
    dataset_size: int = 0
    save_every: int = 5 #how often to save, in iterations
    min_difficulty_time: int = 10 #at least how many times example needs to have been classified correctly to be
                                  #considered learned. Note that this is not in iterations, it just refers to
                                  #the iterations mod save_every (how often we actually log this metric). If
                                  #save_every = 1 then it would just mean iterations.

    def __post_init__(self):

        if self.dataset_size <= 0 or self.batch_size <= 0:
            raise ValueError('Invalid dataset size...')
            sys.exit(0)

        self.train_accuracy = {}
        self.test_accuracy = {}
        self.model_outputs = {}
        self.classification = torch.zeros(dataset_size)

        self._iteration = 0
        self._epoch = 0

        self.difficulty_mask = {}

    def description(self) -> str:
        return 'Metric to log train and test accuracy.'

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

    def pre_iteration(self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader = None) -> None:
        if not dataloader:
            raise ValueError('Specify a valid dataset to get accuracy metric!')
            sys.exit(1)

        if self._iteration % self.save_every != 0:
            return

        classification = torch.zeros(len(dataloader))
        class_idx = 0 #we need an index to track which example we're looking at in the full dataset
        model.eval()
        for batch_idx, batch in enumerate(dataloader):
            x, y = batch
            outputs = model(x.cuda())
            self.model_outputs[self._epoch, self._iteration] = outputs.detach()
            self.train_accuracy[self._epoch, self._iteration] = y.eq(outputs.detach().argmax(dim=1).cpu()).float().mean()

            for ex_idx, ex in enumerate(batch):
                classification[batch_idx + class_idx] = (outputs[ex_idx].detach() == y[ex_idx].detach())
                class_idx+=1
                
            self.classification[self._epoch, self._iteration] = classification
        

        model.train()

        self.create_difficulty_mask()

    def post_iteration(self) -> None:
        """Functions to execute during once batch is loaded and after optimizer step."""
        self._iteration += 1

    def end_epoch(self) -> None:
        """Functions to execute at the end of an epoch."""
        self._epoch += 1

    def end_training(self) -> None:
        utils.save.save_object(object = self.train_accuracy,
                         output_location = self.output_location,
                         object_name = 'IterationAccuracy')

        utils.save.save_object(object = self.model_outputs,
                         output_location = self.output_location,
                         object_name = 'ModelOutputs')

        utils.save.save_object(object = self.classification,
                         output_location = self.output_location,
                         object_name = 'DatasetClassification')

        utils.save.save_object(object = self.difficulty_mask,
                         output_location = self.output_location,
                         object_name = 'DifficultyMask')

    def create_difficulty_mask(self) -> None:
        self.difficulty_mask[self._epoch, self._iteration] = tracking.difficulty_mask.DifficultyMask(dataset_size = self.dataset_size)
        self.difficulty_mask[self._epoch, self._iteration].set_mask_on(classifications = self.classification[self._epoch, self._iteration])