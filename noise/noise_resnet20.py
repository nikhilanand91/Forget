from noise.noise_hparams import NoiseHParams
from base.noiseloop import NoiseLoop

import torch

class NoiseResnet20(NoiseLoop):

	def __init__(self, noise_hparams: NoiseHParams):
		pass

	@staticmethod
	def description():
		return 'A class to add noise to ResNet20 models.'

	def noise_fcn(self):
		pass

	def add_noise(self, model: torch.nn.Module):
		pass

	def loop(self):
		print(f'Looping!')
		pass