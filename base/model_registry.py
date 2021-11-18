from training.train_hparams import TrainHParams
from models.cifar_resnet import CifarResnet
from torch import nn, optim

def get_model(model_name: str):
    if model_name == 'cifar_resnet_20':
        return CifarResnet.get_resnet_20()

def get_optimizer(hparams: TrainHParams, model: nn.Module):
    if hparams.optim == 'SGD' or hparams.optim == 'sgd':
        return optim.SGD(params = model.parameters(), lr = hparams.lr, momentum = hparams.momentum)

def get_loss(loss_name: str):
    if loss_name == 'CrossEntropy':
        return nn.CrossEntropyLoss()