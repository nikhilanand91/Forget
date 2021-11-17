"""
Basic utility functions for printing.
"""

from training.hparams import TrainHParams


def print_train_hparams(train_hparams: TrainHParams) -> None:
    print('\n'+'-'*50)
    print(' '*10 + ' Training hyperparameters:')
    print('-'*50)
    print(f' '*5 + f'Model name: {train_hparams.model}')
    print(f' '*5 + f'Datset: {train_hparams.dataset}')
    print(f' '*5 + f'Output directory: {train_hparams.output_location}')
    print(f' '*5 + f'Optimizer: {train_hparams.optim}')
    print(f' '*5 + f'Learning rate: {train_hparams.lr}')
    print(f' '*5 + f'Momentum: {train_hparams.momentum}')
    print(f' '*5 + f'No. of train epochs: {train_hparams.num_ep}ep')
    print(f' '*5 + f'Checkpoint every: {train_hparams.chkpoint_step}ep')
    print('-'*50 + '\n')

    """
    just use this instead later:
    
    for field in train_hparams.__dataclass_fields__:
        value = getattr(train_hparams, field)
        print(field, value)
    """

    return

def save_to_file(output_location: str) -> None:
    with open(output_location + '/test.csv', 'w', newline='') as csvfile:
        param_writer = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)

        param_writer.writerow(['Spam'] * 5 + ['Baked Beans'])
        param_writer.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])