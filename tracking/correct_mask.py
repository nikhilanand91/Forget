from base.mask import Mask
from datasets.datasets import Dataset

from typing import List
import torch

class CorrectMask(Mask):

	def __init__(self, dataset_size):
		self.dataset_size = dataset_size
		self.mask = torch.zeros(dataset_size)

	def set_mask_on(classification, ordering):
		for classified_correctly, order in zip(classification, ordering):
			if classified_correctly:
				self.mask[order] = 1

	def apply_mask(dataset: Dataset):
		dataset_to_apply_on = dataset.get_dataset()
		return torch.utils.data.Subset(dataset_to_apply_on, self.mask)