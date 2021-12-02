from base.mask import Mask
from datasets.datasets import Dataset

from typing import List
import torch

class CorrectMask(Mask):

	def __init__(self, dataset_size):
		self.dataset_size = dataset_size
		self.mask = torch.zeros(dataset_size)

	def set_mask_on(classification, ordering):
		for classification, order in zip(classification, ordering):
			if classification:
				self.mask[order] = 1

	def apply_mask(Dataset):
		pass