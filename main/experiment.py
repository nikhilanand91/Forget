import sys
from pathlib import Path
import configparser
import parser
import numpy


class experiment:
    """
    the experiment should call on config.py to get info and create the appropriate directories
    based on the contents of config.ini file. It should then be divided into two steps:
    1. Pretraining (e.g. load model from OpenLTH)
    2. Training (for each job, pass models onto trainer.py which trains it and stores the data)
    """
    
    def __init__(self, config_file = "default_config.ini"):
        #pretraining step:
        #first add OpenLTH to the path
        parent_dir_path = Path(Path().absolute()).parent
        sys.path.append(str(parent_dir_path) + 'open_lth-master/')
        
        from open_lth.foundations import hparams
        from open_lth.models import registry

        #get config files from parser
        self.reader = parser.readConfig(config_file)

        #get the number of models to train
        self.num_models = int(self.reader.exp_info["number of models"])
        self.num_jobs = int(self.reader.exp_info["number of jobs"])

        #number of models to train per job
        if self.num_models % self.num_jobs == 0:
            self.num_train_per_job = numpy.full(self.num_jobs, self.num_models/self.num_jobs)
        else:
            self.num_train_per_job = numpy.full(self.num_jobs - 1, int(self.num_models/self.num_jobs))
            self.num_train_per_job = numpy.append(self.num_train_per_job, self.num_models % self.num_jobs) #check this

        #and for each job, pass models onto trainer
        for num_job in self.num_train_per_job:
            for model_no in range(int(num_job)):
                print(model_no)