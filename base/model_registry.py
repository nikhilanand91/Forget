from training.train_hparams import TrainHParams
from models.cifar_resnet import CifarResnet
from torch import nn, optim

def get_model(hparams: TrainHParams):
    if hparams.model.startsWith('cifar_resnet_'):
        return CifarResnet.get_model(hparams)

def get_optimizer(hparams: TrainHParams, model: Model):
    if hparams.optim == 'SGD' or hparams.optim == 'sgd':
        return optim.SGD(params = model.parameters(), lr = hparams.lr, momentum = hparams.momentum)