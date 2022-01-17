import argparse
from dataclasses import dataclass
from pathlib import Path
import os.path

from base.runner import Runner
from noise.noise_hparams import NoiseHParams
import noise.noise_resnet20

@dataclass
class NoiseRunner(Runner):
    """Runner for noise experiments."""
    noise_hparams: NoiseHParams

    @staticmethod
    def description():
        return "Runner to apply noise to ResNet models."

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        allowed_params, param_desc = NoiseHParams().allowed_params, \
                                     NoiseHParams().param_desc

        for param in allowed_params.keys():
            try:
                parser.add_argument(param, 
                                    type = allowed_params[param], 
                                    help = param_desc[param])
            except KeyError as err:
                print('Make sure hyperparameters and their descriptions match! Missing keys: \n')
                print(err)

    @staticmethod
    def create_from_args(args: argparse.Namespace) -> 'NoiseRunner':
        noise_hparams = NoiseHParams()

        #set hparams based on what is supplied from commandline
        for arg in vars(args):
            arg_value = getattr(args, arg)
            if arg_value != None and arg in dir(noise_hparams):
                setattr(noise_hparams, arg, arg_value)

        return NoiseRunner(noise_hparams)

    def make_output_directory(self) -> None:
        if not os.path.exists(self.noise_hparams.model_directory):
            raise ValueError(f'Set a valid model directory! Right now it is: {self.noise_hparams.model_directory}')
            sys.exit(1)

        Path(self.noise_hparams.model_directory + 'noise/').mkdir(parents = True, exist_ok = True)

    def display_output_location(self):
        print(f'Output directory: {self.noise_hparams.model_directory}')

    def run(self):
        #kick off job to add noise to models
        noise_resnet20 = noise.noise_resnet20.NoiseResnet20(noise_hparams = self.noise_hparams)
        noise_resnet20.loop()