from training.hparams import TrainHParams
from utils.print_utils import print_train_hparams
from datasets.datasets import Dataset
from models.models import cifar_resnet_20

def train_loop(train_hparams: TrainHParams):
    """
    Main training loop code.
    """
    print_train_hparams(train_hparams)
    
    #Get model and dataset.
    dataset = Dataset(dataset_name = train_hparams.dataset, output_location = train_hparams.output_location).get_dataset()
    model = Model(train_hparams = train_hparams)
    optim = model.get_optimizer()
