from dataclasses import dataclass
from base.hparams import HParams

@dataclass
class TrainHParams(HParams):
    """This class contains the training hyperparameter details."""

    allowed_params = {'--model': str, '--dataset': str, '--output_location': str, 
                      '--optim': str, '--loss': str, '--lr': float, '--momentum': float, '--num_ep': int,
                      '--chkpoint_step': int, '--batch_size': int, '--rand_batches': bool}
    param_desc = {'--model': 'the name of the model to train',
                  '--dataset': 'the name of the dataset to train on',
                  '--output_location': 'where to store the checkpointed models',
                  '--optim': 'which optimizer to use (default SGD)',
                  '--loss': 'loss function do use (default cross entropy)',
                  '--lr': 'learning rate (default 1e-3)',
                  '--momentum': 'momentum (default 0.9)',
                  '--num_ep': 'number of epochs to train (default 10ep)',
                  '--chkpoint_step': 'how often to save model in iterations (default every 100it)',
                  '--batch_size': 'number of examples to include in each batch of training (default 128)',
                  '--rand_batches': 'randomize the order of batches in each epoch (default false)'}

    model: str = ''
    dataset: str = ''
    output_location: str = ''

    optim: str = 'SGD'
    loss: str = 'CrossEntropy'
    lr: float = 1e-3
    momentum: float = 0.9
    num_ep: int = 10
    chkpoint_step: int = 100
    batch_size: int = 128
    rand_batches: bool = False