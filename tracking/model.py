from dataclasses import dataclass
import torch

from base.metriclogger import MetricLogger

import utils.save

@dataclass
class Model(MetricLogger):

    chkpoint_step: int = 100 #how often to save the model, in iterations
    output_location: str = '/'

    def __post_init__(self):
        self._iteration = 0
        self._epoch = 0

    def description(self) -> str:
        return 'Metric to log model states.'

    def needs(self) -> None:
        print(f'Pre iteration function needs: model.')

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

    def pre_iteration(self, model: torch.nn.Module) -> None:
        """Functions to execute in the pre-iteration step."""
        if self._iteration % self.chkpoint_step != 0:
            return
        utils.save.save_model(output_location = self.output_location, model = model, iteration = self._iteration)
        

    def post_iteration(self) -> None:
        """Functions to execute during once batch is loaded and after optimizer step."""
        self._iteration += 1

    def end_epoch(self) -> None:
        """Functions to execute at the end of an epoch."""
        self._epoch += 1

    def end_training(self) -> None:
        """Functions to execute at the end of training."""
        pass