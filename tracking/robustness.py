from tracking.metriclogger import MetricLogger

class Robustness(MetricLogger):
	
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