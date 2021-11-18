from training.train_hparams import TrainHParams
from utils.print_utils import print_train_hparams
from base import model_registry
from datasets.datasets import Dataset

from torch.utils.data import DataLoader

def train_loop(train_hparams: TrainHParams):
    """
    Main training loop code.
    """
    print_train_hparams(train_hparams)
    
    #Get dataset, model, loss, and optimizer.
    dataset = Dataset().get_dataset(dataset_name = train_hparams.dataset, \
                                           output_location = train_hparams.output_location)
    dataloader = DataLoader(dataset = dataset, batch_size = train_hparams.batch_size)

    model = model_registry.get_model(model_name = train_hparams.model)
    loss = model_registry.get_loss(loss_name = train_hparams.loss)
    optim = model_registry.get_optimizer(hparams = train_hparams, model = model)

    #define accuracy and loss
    batch_accuracy = list()

    #Loop over epochs and batches.
    model.train()
    for epoch in range(train_hparams.num_ep):
        for batch in dataloader:
            x, y = batch
            x = x.cuda()
            outputs = model(x)

            J = loss(outputs, y.cuda())
            model.zero_grad()
            J.backward()
            optim.step()

            batch_accuracy.append(y.eq(outputs.detach().argmax(dim=1).cpu()).float().mean())
        print(batch_accuracy)