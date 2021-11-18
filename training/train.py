from training.train_hparams import TrainHParams
from utils.print_utils import print_train_hparams
from base import model_registry
from datasets.datasets import Dataset

def train_loop(train_hparams: TrainHParams):
    """
    Main training loop code.
    """
    print_train_hparams(train_hparams)
    
    #Get model and dataset.
    dataset = Dataset().get_dataset(dataset_name = train_hparams.dataset, \
                                           output_location = train_hparams.output_location)
    model = model_registry.get_model(model_name = train_hparams.model)
    optim = model_registry.get_optimizer(hparams = train_hparams, model = model)

    
