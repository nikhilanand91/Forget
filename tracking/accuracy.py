from dataclasses import dataclass
from collections.abc import Iterable
from typing import List
import torch

from base.metriclogger import MetricLogger
import base.dataset
import tracking.correctness_mask

import utils.save

@dataclass
class Accuracy(MetricLogger):

    replicate: int  #which replicate model we are tracking
    dataset_size: int = 0
    output_location: str = '/'
    save_every: int = 100 #how often to save, in iterations
    min_learned_time: int = 100 #Learned examples at iteration t are those
                                #that are classified correctly for all t' >= t. Note that we require t to be
                                #bounded by t <= maximum # of iterations trained - min_learned_time.

    def __post_init__(self):

        if self.dataset_size <= 0:
            raise ValueError('Invalid dataset size...')
            sys.exit(0)

        self.train_accuracy = {}
        self.test_accuracy = {}
        self.model_outputs = {}
        self.classification = {}

        self._iteration = 0
        self._epoch = 0

        self.correctness_mask = {}
        self.learned_examples = {}
        #self.learned_mask = tracking.correctness_mask.CorrectnessMask(dataset_size = self.dataset_size)

    def description(self) -> str:
        return 'Metric to log train and test accuracy.'

    def needs(self) -> None:
        print(f'Pre iteration functions need: model, dataloader.')

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

        classification = torch.zeros(self.dataset_size)
        model.eval()
        for batch_idx, batch in enumerate(dataloader):
            x, y = batch
            outputs = model(x.cuda())
            self.model_outputs[self._iteration] = outputs.detach()
            self.train_accuracy[self._iteration] = y.eq(outputs.detach().argmax(dim=1).cpu()).float().mean()

            classification[batch_idx: batch_idx+len(outputs)] = y.eq(outputs.detach().argmax(dim=1).cpu())
                
        self.classification[self._iteration] = classification
        

        model.train()

        self.create_correctness_mask()
        

    def post_iteration(self) -> None:
        """Functions to execute during once batch is loaded and after optimizer step."""
        self._iteration += 1

    def end_epoch(self) -> None:
        """Functions to execute at the end of an epoch."""
        self._epoch += 1

    def end_training(self) -> None:
        """Functions to execute at the end of training."""
        self.create_learned_mask()

        utils.save.save_object(object = self.train_accuracy,
                         output_location = self.output_location,
                         object_name = 'IterationAccuracyRep' + str(self.replicate))

        utils.save.save_object(object = self.model_outputs,
                         output_location = self.output_location,
                         object_name = 'ModelOutputsRep' + str(self.replicate))

        utils.save.save_object(object = self.classification,
                         output_location = self.output_location,
                         object_name = 'DatasetClassificationRep' + str(self.replicate))

        utils.save.save_object(object = self.correctness_mask,
                         output_location = self.output_location,
                         object_name = 'CorrectMaskRep' + str(self.replicate))

        utils.save.save_object(object = self.learned_examples,
                         output_location = self.output_location,
                         object_name = 'LearnedExamplesRep' + str(self.replicate))

    def create_correctness_mask(self) -> None:
        correctness_mask = tracking.correctness_mask.CorrectnessMask(dataset_size = self.dataset_size)
        correctness_mask.set_mask_on(classifications = self.classification[self._iteration])
        self.correctness_mask[self._iteration] = correctness_mask.idx

    def create_learned_mask(self) -> None:
        """Create a dictionary of learned examples."""
        keys = self.correctness_mask.keys()
        for ex_idx in range(self.dataset_size):
            time_learned = float('inf')
            for iteration in range(self._iteration, -1, -1):
                if iteration in keys and self.correctness_mask[iteration][ex_idx]:
                    time_learned = iteration
                elif iteration not in keys:
                    continue
                else:
                    break

            if self._iteration - time_learned >= self.min_learned_time:
                self.learned_examples[ex_idx] = time_learned
            else:
                self.learned_examples[ex_idx] = float('inf')