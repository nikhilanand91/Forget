#next step is a trainer loop with a decorator that takes info from
#the reader and sets up the appropriate training loop

import torch
from torch import nn, optim
from torchvision import datasets, transforms, utils
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt

loss_A = nn.CrossEntropyLoss()
optimizer_A = optim.SGD(model_A.parameters(), lr=1e-3, momentum=0.9)

class train:
    def __init__(self, model, job_info):
        #list of datasets that trainer knows about
        self.dataset_names = ['cifar10']
        self.num_epochs = job_info["num epochs"]

        if job_info["model parameters"] == "default":
            self.optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
            self.loss = nn.CrossEntropyLoss()
        else:
            pass #to add this functionality, custom loss and optimizer

        if job_info["save models"] == "true" or job_info["save models"] == "True"
            self.save_model = True
        else:
            self.save_model = False

        data_idx = [i for i in range(len(self.dataset_names)) if self.dataset_names[i]==job_info["dataset"]]
        self.data_loader = self.getTrainDataset(data_idx)

        if job_info["dataset params"] == "default":
            self.batch_size = 128
        else:
            pass #to add, custom dataset batch size, num workers, etc.

        if job_info["measure forget"] == "true" or job_info["measure forget"] == "True":
            self.forget_msrmt = measureForget(self.num_epochs, num_batches = len(self.data_loader), batch_size=self.batch_size)

        if job_info["track correct examples"] == "true" or job_info["track correct examples"] == "True":
            self.track_correct_ex = True
        
        if job_info["storage directory"] == "default":
            self.store_directory = ""
        else:
            pass #to add..

    
    def getTrainDataset(self, data_idx): #option to change batch size?
        if data_idx == 0:
            train_dataset = datasets.CIFAR10('/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))
            return DataLoader(train_dataset, batch_size=self.batch_size, num_workers = 0)

    def trainLoop(self):
        losses = list()
        accuracies = list()
        epochs = list()
        model.train()

        for epoch in range(num_epochs):
            batch_loss = list()
            batch_acc = list()

            for batch in train_set:
                x,y = batch
                x=x.cuda()
                logits = model(x)
                if forget_msrmt != None:
                    forget_msrmt.trackForgettableExamples(logits.detach(), y.detach())

                J = loss(logits, y.cuda())
                model.zero_grad()
                J.backward()
                optimizer.step()

                batch_loss.append(J.item())
                batch_acc.append(y.eq(logits.detach().argmax(dim=1).cpu()).float().mean())

                if forget_msrmt != None:
                    forget_msrmt.incrementTrainBatch()
            
            if forget_msrmt != None:
                forget_msrmt.resetTrainBatchTracker()
            
            accuracies.append(torch.tensor(batch_acc).mean())
    
    def save_model(self):

