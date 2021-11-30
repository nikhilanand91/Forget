from torch.utils.data import Sampler
import random

class RandomSampler(Sampler):
	def __init__(self, data_source):
		self.data_source = data_source
		self.shuffle()
		self.idx = None

	def shuffle(self):
		self.idx = list(range(len(self.data_source)))
		self.seed = random.randint(0, 2**32-1)
		random.Random(self.seed).shuffle(self.idx)
		return self.idx
	
	def get_order(self):
		return self.idx

	def __iter__(self):
		return iter(self.idx)

	def __len__(self):
		return len(self.data_source)