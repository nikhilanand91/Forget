from training.train_hparams import TrainHParams
from base.trainloop import TrainLoop

import base.model_registry
import base.dataset_registry
import tracking.robustness
import tracking.accuracy

import torch

class TrainRobust(TrainLoop):

	def __init__(self, train_hparams: TrainHParams):
		self.dataset_object = base.dataset_registry.get_dataset_object(dataset_name = train_hparams.dataset,
                                                         save_location = train_hparams.output_location,
                                                         shuffle = train_hparams.rand_batches)
    
	    self.dataset = self.dataset_object.get_dataset() #get the actual dataset
	    self.dataset_object.get_sampler() #set the sampler
	    self.dataloader = self.dataset_object.get_dataloader(batch_size = train_hparams.batch_size)
	    
	    self.model = base.model_registry.get_model(model_name = train_hparams.model).cuda()
	    self.loss = base.model_registry.get_loss(loss_name = train_hparams.loss)
	    self.optim = base.model_registry.get_optimizer(hparams = train_hparams, model = model)


	    self.num_ep = train_hparams.num_ep
	    self.batch_size = train_hparams.batch_size

	    #define accuracy and loss
	    self.batch_accuracy = list()

	    #define which metrics we're logging
	   	self.accuracy_metric = tracking.accuracy.Accuracy(dataset_size = len(self.dataset),
	   													  output_location = train_hparams.output_location)


	@staticmethod
	def description():
		return 'A training loop to measure robustness and accuracy statistics.'


	def loop(self):
		self.accuracy_metric.pre_training()

		self.model.train()
	    for epoch in range(self.num_ep):
	        
	        self.accuracy_metric.start_epoch()

	        if train_hparams.rand_batches:
	            sampler = dataset_object.get_sampler() #with randomized examples, we reset the sampler after each epoch
	            self.dataloader = dataset_object.get_dataloader(batch_size = self.batch_size)

	        for batch in self.dataloader:
	            x, y = batch

	            x = x.cuda()
	            outputs = self.model(x)

	            self.accuracy_metric.pre_iteration(model = self.model,
	                                               dataloader = self.dataloader)



	            J = self.loss(outputs, y.cuda())
	            self.model.zero_grad()
	            J.backward()
	            self.optim.step()

	            
	            self.accuracy_metric.post_iteration()

	            batch_accuracy.append(y.eq(outputs.detach().argmax(dim=1).cpu()).float().mean())
	        print(torch.tensor(batch_accuracy).mean())

	        
	        self.accuracy_metric.end_epoch()

	    self.accuracy_metric.end_training()