import abc

class TrainLoop(abc.ABC):
	"""
	An instance of a training loop. It is useful to make this
	an abstract base class because depending on the metrics we want
	to log, the training loop will look very different. It is
	therefore useful to have a blueprint and use a different
	instance for different metrics (otherwise the loop will
	become way too long/complicated.)
	"""

	@staticmethod
	@abc.abstractmethod
	def description():
		"""A description of this trianing loop."""
		pass

	@abc.abstractmethod
	def loop(self):
		"""The loop itself."""
		pass