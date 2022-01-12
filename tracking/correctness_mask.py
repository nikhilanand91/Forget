import torch

from base.mask import Mask
import base.dataset

class CorrectnessMask(Mask):

	def __init__(self, dataset_size):
		self.dataset_size = dataset_size

	def set_mask_on(classifications: torch.Tensor):
		self.idx = classifications

	def apply_mask(dataset: base.dataset.Dataset):
		dataset_to_apply_on = dataset.get_dataset()
		return torch.utils.data.Subset(dataset_to_apply_on, self.idx)