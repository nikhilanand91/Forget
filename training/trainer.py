import torch
from torch import nn, optim
from torchvision import datasets, transforms, utils
from torch.utils.data import random_split, DataLoader
from pathlib import Path
import matplotlib.pyplot as plt

from Forget.training  import measureforget

class train:
    def __init__(self, model, exp_info, job_info, job_idx, model_idx): #job_idx, model_idx should be a unique modifier that indexes the job, model
        #structure of directory is eg ../jobs/job1/model1/
        #idx here would be '1'

        #list of datasets that trainer knows about
        parent_dir_path = Path(Path().absolute()).parent

        self.dataset_names = ['cifar10']
        self.num_epochs = int(job_info["num epochs"])
        self.save_every = int(job_info["save every"])
        if exp_info["storage directory"] == "default":
            self.exp_directory = str(parent_dir_path) + exp_info["name"] + "/"
        else:
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

        #print(job_info["dataset params"])
        if job_info["dataset params"] == "default":
            self.batch_size = 128 #note that this also gets passed to measureForget
        else:
            pass #to add, custom dataset batch size, num workers, etc.
        
        data_idx = [i for i in range(len(self.dataset_names)) if self.dataset_names[i]==job_info["dataset"]][0]
        self.data_loader = self.getTrainDataset(data_idx)

        if job_info["measure forget"] == "true" or job_info["measure forget"] == "True":
            self.forget_flag = True
            self.forget_msrmt = measureforget.measureForget(self.num_epochs, num_batches = len(self.data_loader), batch_size=self.batch_size)
        else:
            self.forget_msrmt = None

        if job_info["track correct examples"] == "true" or job_info["track correct examples"] == "True":
            self.track_correct_ex = True
        
        if job_info["storage directory"] == "default":
            self.store_directory = self.exp_directory + "Job " + str(job_idx+1) + "/" + "model" + str(model_idx) + "/"
        else:
            pass #to add..
        
        #self.trainLoop(model) #train the model
    
    def getTrainDataset(self, data_idx): #option to change batch size?
        print(f"Loading train dataset {self.dataset_names[data_idx]}... batch size {self.batch_size}")
        if data_idx == 0:
            train_dataset = datasets.CIFAR10('/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))
            return DataLoader(train_dataset, batch_size=self.batch_size, num_workers = 0)

    def trainLoop(self, model):
        losses = list()
        accuracies = list()
        epochs = list()

        for epoch in range(self.num_epochs):
            batch_loss = list()
            batch_acc = list()

            model.train()
            for batch in self.data_loader:
                x,y = batch
                x=x.cuda()
                logits = model(x)

                if self.forget_flag: #eventually should change forget class to have wrapper instead of these flags.
                    self.forget_msrmt.trackForgettableExamples(logits.detach(), y.detach())

                J = self.loss(logits, y.cuda())
                model.zero_grad()
                J.backward()
                self.optimizer.step()

                batch_loss.append(J.item())
                batch_acc.append(y.eq(logits.detach().argmax(dim=1).cpu()).float().mean())

                if self.forget_flag:
                    self.forget_msrmt.incrementTrainBatch()
            
            if self.forget_flag:
                self.forget_msrmt.resetTrainBatchTracker()

            if self.track_correct_ex:
                model.eval()
                for batch in self.data_loader:
                    x, y = batch
                    x = x.cuda()
                    with torch.no_grad():
                        logits_prime = model(x.detach())
        
                    self.forget_msrmt.trackCorrectExamples(logits_prime.detach(), y.detach())
                    self.forget_msrmt.incrementClassifyBatch()
    
                self.forget_msrmt.resetClassifyBatchTracker()

            if (epoch+1) % self.save_every == 0:
                self.save_model_data(model, epoch, torch.tensor(batch_loss).mean(), torch.tensor(batch_acc).mean()) #, torch.tensor(batch_loss).mean())
                self.save_data()
            
            accuracies.append(torch.tensor(batch_acc).mean())
            if self.forget_flag:
                self.forget_msrmt.incrementTrainIter()

        if self.forget_flag:
            self.forget_msrmt.resetTrainIter()
        
        model.eval()
        self.clean(model)
    
    def save_model_data(self, model, epoch, loss, accuracy):
        torch.save({
           'epoch': epoch+1,
           'model_state_dict': model.state_dict(),
           'optimizer_state_dict': self.optimizer.state_dict(),
           'loss': loss,
           'train accuracy': accuracy,
           }, self.store_directory + "epoch=" + str(epoch+1) + ".pt")
        
    def save_data(self):
        if self.forget_flag:
            self.forget_msrmt.saveForget(self.store_directory)
        if self.track_correct_ex:
            self.forget_msrmt.saveCorrect(self.store_directory)

        #to add: save accuracies

    def clean(self, model):
        del model
        del self.forget_msrmt

        #after training, clean caches,..
