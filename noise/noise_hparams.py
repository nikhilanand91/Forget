from dataclasses import dataclass
from base.hparams import HParams

@dataclass
class NoiseHParams(HParams):
    """This class contains the training hyperparameter details."""

    allowed_params = {'--model_directory': str, '--iteration': int, 'replicates': int,
                      '--noise_min': int, '--noise_max': int, '--num_samples': int}
    param_desc = {'--model_directory': 'directory where trained models are located',
                  '--replicates:': 'which replicates to do noise experiments on (e.g, 4 means replicates 0, 1, ... , 4)',
                  '--iter': 'the iteration of model to add noise to',
                  '--noise_min': 'minimum level of noise (default 0)',
                  '--noise_max': 'maximum level of noise (default 1)',
                  '--num_samples': 'number of noise samples (default 100)'}

    model_directory: str
    iteration: int
    replicates: int
    noise_min: int = 0
    noise_max: int = 1
    num_samples: int = 100