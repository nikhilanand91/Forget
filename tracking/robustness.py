from tracking.metriclogger import MetricLogger
from dataclasses import dataclass
from tracking.robust_mask import RobustMask

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

        self.robustness_stats = 

	
	def description() -> str:
    	return 'Metric to log robustness statistics.'

    def pre_training() -> None:
        """Functions to execute before training loop starts."""

    def start_epoch() -> None:
        """Functions to execute at the start of each epoch but before we load a batch."""

    def pre_iteration() -> None:
        """Functions to execute during once batch is loaded but before optimizer step."""

    def post_iteration() -> None:
        """Functions to execute during once batch is loaded and after optimizer step."""

    def end_epoch() -> None:
        """Functions to execute at the end of an epoch."""