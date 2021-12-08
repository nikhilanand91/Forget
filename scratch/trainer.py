from dataclasses import dataclass

from training.train_hparams import TrainHParams
from utils.print_utils import print_train_hparams
from base import model_registry, dataset_registry

@dataclass
class Trainer:
    """
    The trainer class which provides the components necessary for training,
    including the training loop.
    """

    train_hparams: TrainHParams

    def __post_init__(self):
        print_train_hparams(train_hparams)

        self.dataset_object = dataset_registry.get_dataset_object(dataset_name = train_hparams.dataset,
                                                                  save_location = train_hparams.output_location,
                                                                  shuffle = train_hparams.rand_batches)
    
        self.dataset = dataset_object.get_dataset()
        self.dataset_object.get_sampler() #set the sampler
        self.dataloader = dataset_object.get_dataloader(batch_size = train_hparams.batch_size)
        
        self.model = model_registry.get_model(model_name = train_hparams.model).cuda()
        self.loss = model_registry.get_loss(loss_name = train_hparams.loss)
        self.optim = model_registry.get_optimizer(hparams = train_hparams, model = model)


    def get_batch(self):
        return next(self.dataloader)

    def get_order(self):
        