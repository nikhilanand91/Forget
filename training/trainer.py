import torch
from torch import nn, optim
from torchvision import datasets, transforms, utils
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt

from measureforget import measureForget

class train:
    def __init__(self, model, exp_info, job_info, job_idx, model_idx): #job_idx, model_idx should be a unique modifier that indexes the job, model
        #structure of directory is eg ../jobs/job1/model1/
        #idx here would be '1'

        #list of datasets that trainer knows about
        self.dataset_names = ['cifar10']
        self.num_epochs = int(job_info["num epochs"])
        self.exp_directory = exp_info["storage directory"]

        if job_info["model parameters"] == "default":
            self.optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
            self.loss = nn.CrossEntropyLoss()
        else:
            pass #to add this functionality, custom loss and optimizer

        if job_info["save models"] == "true" or job_info["save models"] == "True":
            self.save_model = True
        else:
            self.save_model = False

        data_idx = [i for i in range(len(self.dataset_names)) if self.dataset_names[i]==job_info["dataset"]]
        self.data_loader = self.getTrainDataset(data_idx)

        if job_info["dataset params"] == "default":
            self.batch_size = 128 #note that this also gets passed to measureForget
        else:
            pass #to add, custom dataset batch size, num workers, etc.

        if job_info["measure forget"] == "true" or job_info["measure forget"] == "True":
            self.forget_flag = True
            self.forget_msrmt = measureForget(self.num_epochs, num_batches = self.batch_size, batch_size=self.batch_size)

        if job_info["track correct examples"] == "true" or job_info["track correct examples"] == "True":
            self.track_correct_ex = True
        
        if job_info["storage directory"] == "default":
            self.store_directory = self.exp_directory + "/job" + str(job_idx) + "/" + "model" + str(model_idx) + "/"
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

                if self.forget_flag: #eventually should change forget class to have wrapper instead of these flags.
                    forget_msrmt.trackForgettableExamples(logits.detach(), y.detach())

                J = loss(logits, y.cuda())
                model.zero_grad()
                J.backward()
                optimizer.step()

                batch_loss.append(J.item())
                batch_acc.append(y.eq(logits.detach().argmax(dim=1).cpu()).float().mean())

                if self.forget_flag:
                    forget_msrmt.incrementTrainBatch()
            
            if self.forget_flag:
                forget_msrmt.resetTrainBatchTracker()
            
            accuracies.append(torch.tensor(batch_acc).mean())
            if self.forget_flag:
                forget_msrmt.incrementTrainIter()
        
        model.eval()
    
    def save_model(self):
        pass
        #model,

    def save_data(self):
        pass
        #save accuracies, epoch list

    def clean(self):
        pass
        #after training, clean caches,