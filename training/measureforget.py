from dataclasses import dataclass
import torch

@dataclass
class measureForget:
    nb_epochs: int
    num_batches: int
    batch_size: int

    forgetStatistics: torch.zeros(nb_epochs, num_batches, batch_size)
    correctStatistics: torch.zeros(nb_epochs, num_batches, batch_size)
    a_i: torch.zeros(nb_epochs, num_batches, batch_size) #measures if correctly classified or not
    softmaxfunc: nn.Softmax(dim=1)
    train_batch_tracker: 0
    classify_batch_tracker: 0
    train_iteration: 0
    num_ep: nb_epochs
    num_btchs: num_batches
    btc_size: batch_size

    #track examples check if examples in a batch were correctly classified
    #before and aren't classified correctly now (where before and now refer
    #to subsequent training iterations)
    def trackForgettableExamples(self, batch_model_output, labels):
        if self.train_iteration < 1:
            pass
        else:
            counter = 0
            for logit in batch_model_output:
                if torch.argmax(logit) == labels[counter]:
                    self.a_i[self.train_iteration, self.train_batch_tracker, counter] = 1
                else:
                    self.a_i[self.train_iteration, self.train_batch_tracker, counter] = 0
            
                if self.a_i[self.train_iteration, self.train_batch_tracker, counter] < self.a_i[self.train_iteration-1, self.train_batch_tracker, counter]:
                    self.forgetStatistics[self.train_iteration, self.train_batch_tracker, counter]+=1
            
                counter+=1
    
    #track the examples which are classified correctly at any given moment in training
    def trackCorrectExamples(self, batch_model_output, labels):
        for k in range(len(batch_model_output)):
            if torch.argmax(batch_model_output[k]) == labels[k]:
                self.correctStatistics[self.train_iteration, self.classify_batch_tracker, k] = 1
    
    def resetCorrectStatistics(self):
        self.correctStatistics = torch.zeros(self.num_ep, self.num_btchs, self.btc_size)

    def incrementTrainBatch(self):
        self.train_batch_tracker+=1
    
    def incrementClassifyBatch(self):
        self.classify_batch_tracker+=1
    
    def getTrainBatchNumber(self):
        return self.train_batch_tracker
    
    def getClassifyBatchNumber(self):
        return self.classify_batch_tracker

    def getTrainIteration(self):
        return self.train_iteration

    def incrementTrainIter(self):
        self.train_iteration+=1
    
    def decrementTrainIter(self):
        self.train_iteration-=1

    def resetTrainBatchTracker(self):
        self.train_batch_tracker=0
    
    def resetClassifyBatchTracker(self):
        self.classify_batch_tracker=0