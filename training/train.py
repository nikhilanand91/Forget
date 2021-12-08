from training.train_hparams import TrainHParams
from utils.print_utils import print_train_hparams
from base import model_registry, dataset_registry
from tracking.robustness import Robustness

import torch

def train_loop(train_hparams: TrainHParams):
    """
    Main training loop code.
    """
    
    
    #Get dataset, model, loss, and optimizer.
    dataset_object = dataset_registry.get_dataset_object(dataset_name = train_hparams.dataset,
                                                         save_location = train_hparams.output_location,
                                                         shuffle = train_hparams.rand_batches)
    
    dataset = dataset_object.get_dataset()
    dataset_object.get_sampler() #set the sampler
    dataloader = dataset_object.get_dataloader(batch_size = train_hparams.batch_size)
    
    model = model_registry.get_model(model_name = train_hparams.model).cuda()
    loss = model_registry.get_loss(loss_name = train_hparams.loss)
    optim = model_registry.get_optimizer(hparams = train_hparams, model = model)

    #define accuracy and loss
    batch_accuracy = list()

    #define which metrics we're logging
    robustness_metric = Robustness(dataset_size = len(dataset),
                                   batch_size = train_hparams.batch_size,
                                   output_location = train_hparams.output_location)
    #accuracy_metric = Accuracy()

    
    robustness_metric.pre_training()

    #Training loop.
    model.train()
    for epoch in range(train_hparams.num_ep):
        
        robustness_metric.start_epoch()

        if train_hparams.rand_batches:
            dataset_object.get_sampler() #with randomized examples, we reset the sampler after each epoch
            dataloader = dataset_object.get_dataloader(batch_size = train_hparams.batch_size)
            
        ordering = dataset_object.get_order(batch_size = train_hparams.batch_size)

        for batch in dataloader:
            x, y = batch

            x = x.cuda()
            outputs = model(x)

            order = next(ordering)
            robustness_metric.pre_iteration(model_outputs = outputs.detach(),
                                            targets = y.detach(),
                                            ordering = order)

            J = loss(outputs, y.cuda())
            model.zero_grad()
            J.backward()
            optim.step()

            
            robustness_metric.post_iteration()

            batch_accuracy.append(y.eq(outputs.detach().argmax(dim=1).cpu()).float().mean())
        print(torch.tensor(batch_accuracy).mean())

        
        robustness_metric.end_epoch()

    robustness_metric.end_training()