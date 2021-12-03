import abc
from base.dataset import Dataset
from typing import List

class Mask(abc.ABC):
	"""Basic mask object that can be applied to
	a dataset."""

	@abc.abstractmethod
	def set_mask_on(positions: List[int]):
		pass

	@abc.abstractmethod
	def apply_mask(Dataset):
		pass