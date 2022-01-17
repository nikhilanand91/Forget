import abc

class NoiseLoop(abc.ABC):
	"""
	An instance of a noise loop. It is useful to make this
	an abstract base class because depending on the noise we want
	to add, the loop will look very different.
	"""

	@staticmethod
	@abc.abstractmethod
	def description():
		"""A description of this noise loop."""
		pass

	@abc.abstractmethod
	def noise_fcn(self):
		"""The noise distribution e.g. Gaussian, uniform, etc."""
		pass

	@abc.abstractmethod
	def add_noise(self):
		"""Add noise to a specified model."""
		pass

	@abc.abstractmethod
	def loop(self):
		"""Loop over replicates."""
		pass