from dataclasses import dataclass
from base.hparams import HParams

@dataclass
class TrainHParams(HParams):
    """
    This class contains the training hyperparameter details. These include:
    
    -dataset: the name of the dataset
    -model: the model to train
    -output_location: where to store output
    -lr: learning rate (default 1e-3)
    -momentum (defualt 0.9)
    -num_ep: number of epochs to train (default 20)
    -chkpoint_step: how often to save (default every 5ep)
    """

    allowed_params = {'--model': str, '--dataset': str, '--output_location': str, 
                      '--optim': str, '--lr': float, '--momentum': float, '--num_ep': int, 
                      '--chkpoint_step': int, '--batch_size': int}
    param_desc = {'--model': 'the name of the model to train',
                  '--dataset': 'the name of the dataset to train on',
                  '--output_location': 'where to store the checkpointed models',
                  '--optim': 'which optimizer to use',
                  '--lr': 'learning rate (default 1e-3)',
                  '--momentum': 'momentum (default 0.9)',
                  '--num_ep': 'number of epochs to train (default 20ep)',
                  '--chkpoint_step': 'how often to save model (default every 5ep)',
                  '--batch_size': 'number of examples to include in each batch of training'}

    model: str = ''
    dataset: str = ''
    output_location: str = ''

    optim: str = 'SGD'
    lr: float = 1e-3
    momentum: float = 0.9
    num_ep: int = 20
    chkpoint_step: int = 5
    batch_size: int = 32

    def create_from_hparams(self) -> 'TrainHParams':
        pass