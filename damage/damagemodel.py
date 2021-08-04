import torch
import numpy as np

class damageModel:
    def __init__(self, model):
        from foundations import hparams
        from models import registry
        
        model.eval()
        self.model_clones = []
        self.model_state_dict = model.state_dict()
        self.model_hparams = hparams.ModelHparams('cifar_resnet_20', 'kaiming_uniform', 'uniform')

    def addNoise(self, num_points = 200, min_noise = 0., max_noise = 0.1): #returns an array of length num_points, consisting of models increasingly damaged from Gaussian noise with stdev min_noise to max_noise
        from foundations import hparams
        from models import registry

        epsilons = np.linspace(min_noise, max_noise, num_points)
        for i in range(len(epsilons)):
            sys.stdout.write("\r{0}Cloning models...".format("|"*i))
            sys.stdout.flush()
            self.model_clones.append(registry.get(self.model_hparams).cuda())
            self.model_clones[i].load_state_dict(self.model_state_dict)
            

        with torch.no_grad():
            k = 0
            for model in self.model_clones:
                for param in model.parameters():
                    param.multiply_(1+torch.empty(param.size()).cuda().normal_(mean=0,std=epsilons[k]))
                k+=1
            
        for models in self.model_clones:
            models.eval()
        
        return self.model_clones
    
    def getEpsilons(self, num_points = 200, min_noise = 0., max_noise = 0.1):
        return np.linspace(min_noise, max_noise, num_points)