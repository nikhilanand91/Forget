from training.train_hparams import TrainHParams
from utils.print_utils import print_train_hparams
from datasets.datasets import Dataset
from base import model_registry, dataset_registry

def train_loop(train_hparams: TrainHParams):
    """
    Main training loop code.
    """
    print_train_hparams(train_hparams)
    
    #Get model and dataset.
    dataset = Dataset(dataset_name = train_hparams.dataset, output_location = train_hparams.output_location).get_dataset()
    model = model_registry.get_model(hparams = train_hparams)
    optim = model_registry.get_optimizer(hparams = train_hparams, model = model)
