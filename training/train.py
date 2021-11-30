from training.train_hparams import TrainHParams
from utils.print_utils import print_train_hparams
from base import model_registry, dataset_registry
from tracking.robustness import Robustness

import torch

def train_loop(train_hparams: TrainHParams):
    """
    Main training loop code.
    """
    print_train_hparams(train_hparams)
    
    #Get dataset, model, loss, and optimizer.
    dataset = dataset_registry.get_dataset(dataset_name = train_hparams.dataset, \
                                           save_location = train_hparams.output_location)
    dataloader = dataset.get_dataloader(batch_size = train_hparams.batch_size)

    model = model_registry.get_model(model_name = train_hparams.model).cuda()
    loss = model_registry.get_loss(loss_name = train_hparams.loss)
    optim = model_registry.get_optimizer(hparams = train_hparams, model = model)

    #define accuracy and loss
    batch_accuracy = list()

    #define which metrics we're logging
    robustness_metric = Robustness(dataset_size = len(dataset), batch_size = train_hparams.batch_size)
    #accuracy_metric = Accuracy()

    
    robustness_metric.pre_training()

    #Training loop.
    model.train()
    for epoch in range(train_hparams.num_ep):
        
        robustness_metric.start_epoch()

        if train_hparams.rand_batches:
            dataloader = dataset.get_dataloader(batch_size = train_hparams.batch_size, shuffle = True)

        for batch in dataloader:
            x, y = batch

            ordering = dataset.get_relative_ordering(batch = batch, dataloader = dataloader) if train_hparams.rand_batches \
                       else dataset.get_absolute_ordering(batch = batch, dataloader = dataloader)

            x = x.cuda()
            outputs = model(x)

            robustness_metric.pre_iteration(outputs.detach(), y)

            J = loss(outputs, y.cuda())
            model.zero_grad()
            J.backward()
            optim.step()

            
            robustness_metric.post_iteration()

            batch_accuracy.append(y.eq(outputs.detach().argmax(dim=1).cpu()).float().mean())
        print(torch.tensor(batch_accuracy).mean())

        
        metric.end_epoch()