import argparse
from dataclasses import dataclass
from pathlib import Path

from base.runner import Runner
from training.train_hparams import TrainHParams
from training import train

@dataclass
class TrainRunner(Runner):
    """
    Actually executes training. It should call on trainer in train.py
    and pass the object TrainHParams.
    """
    train_params: TrainHParams
    replicates: int = 1

    @staticmethod
    def description():
        return "Train a model."

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        allowed_params, param_desc = TrainHParams().allowed_params, \
                                     TrainHParams().param_desc

        for param in allowed_params.keys():
            try:
                parser.add_argument(param, 
                                    type = allowed_params[param], 
                                    help = param_desc[param])
            except KeyError as err:
                print('Make sure hyperparameters and their descriptions match in hparams.py! Missing keys: \n')
                print(err)

    @staticmethod
    def create_from_args(args: argparse.Namespace) -> 'TrainRunner':
        train_hparams = TrainHParams()

        #set hparams based on what is supplied from commandline
        for arg in vars(args):
            arg_value = getattr(args, arg)
            if arg_value != None and arg in dir(train_hparams):
                setattr(train_hparams, arg, arg_value)

        return TrainRunner(train_hparams)

    def make_output_directory(self) -> None:
        if self.train_params.output_location == None or self.train_params.output_location == '':
            raise ValueError(f'Set a valid output directory! Right now it is: {self.train_params.output_location}')
            sys.exit(1)

        Path(self.train_params.output_location).mkdir(parents = True, exist_ok = True)

    def display_output_location(self):
        print(f'Output directory: {train_params.output_location}')

    def run(self):
        #kick off training loop
        for _ in range(self.replicates):
            train.train_loop(train_hparams = self.train_params)