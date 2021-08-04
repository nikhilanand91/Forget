import torch

class postProcess:
    def __init__(self, num_examples = None):
        self.num_examples = num_examples

    #measure at which noise level an example that's classified correctly becomes misclassifed
    #this function just classifies a dataset given a model
    def classifyDataset(self, data_loader, models):
        if self.num_examples==None:
            raise ValueError("Specify the size of the dataset please.")

        num_models = len(models)
        __catalog = torch.zeros(num_models, self.num_examples)

        num_ex_per_batch = list()
        for batch in data_loader:
            num_ex_per_batch.append(len(batch[0]))
        print(f"Classifying dataset... examples/batch: {num_ex_per_batch}")
        
        modeltrcker = 0
        for model in models:
            model.eval()

            btrkcer = 0
            for batch in data_loader:
                x,y = batch
                x=x.cuda()
                with torch.no_grad():
                    l_A = model(x)
                for k in range(len(l_A)):
                    if torch.argmax(l_A[k]) == y.cuda()[k]:
                        #print(f"{modeltrcker}, {k+sum(num_ex_per_batch[0:btrkcer])}")
                        __catalog[modeltrcker, k+sum(num_ex_per_batch[0:btrkcer])] = 1
                btrkcer+=1        
                
            modeltrcker+=1

        return __catalog
    
    #returns a table consisting of {epsilon at which example was forgotten, times it was forgotten}
    def tabulateNoiseForget(self, catalog, epsilonList, forgetStats):
        epsilonForgotten = list()
        timesForgotten = list()

        for k in range(len(catalog[0])): #go through each example
            idx = next((i for i in range(len(catalog[0:,k])) if catalog[0:,k][i] == 0), None)
            if idx != None:
                epsilonForgotten.append(epsilonList[idx])
                timesForgotten.append(forgetStats[k])
        
        return epsilonForgotten, timesForgotten

    #For a given set of epsilon, forget stats, it scans through and determines the
    #largest N epsilons for a given # of forgotten events before the example was
    #misclassifed
    def findLargestEpsilon(self, epsilonForgotten, timesForgotten, largestN):
        import heapq
        largest_value = int(max(timesForgotten))
        smallest_value = int(min(timesForgotten))

        largest_epsilon = torch.zeros(largest_value-smallest_value+1, largestN)
        largest_forgotten = torch.zeros(largest_value-smallest_value+1, largestN)

        for j in range(smallest_value, largest_value+1):
            idx = [i for i in range(len(timesForgotten)) if timesForgotten[i]==j]
            for k in range(len(timesForgotten)):
                largest_forgotten[j-smallest_value, 0:] = torch.tensor([j]*largestN)
                largest_epsilon[j-smallest_value, 0:] = torch.tensor(heapq.nlargest(largestN,[epsilonForgotten[i] for i in idx]))

        return torch.flatten(largest_epsilon), torch.flatten(largest_forgotten)

