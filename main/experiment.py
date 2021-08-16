import sys
from pathlib import Path
import configparser
from Forget.config import parser
import numpy
from Forget.training import trainer
import os
from Forget.datasets import createforgetdataset
from Forget.damage import damagemodel


class run_experiment:
    """
    the experiment should call on config.py to get info and create the appropriate directories
    based on the contents of config.ini file. It should then be divided into two steps:
    1. Pretraining (e.g. load model from OpenLTH)
    2. Training (for each job, pass models onto trainer.py which trains it and stores the data)
    """
    
    def __init__(self, config_file = os.getcwd()+"Forget/config/default_config.ini"):
        parent_dir_path = os.path.dirname(str(os.path.dirname(os.path.realpath(__file__))))
        sys.path.append(os.getcwd()+"/Forget/open_lth/")
        #sys.path.append(str(parent_dir_path) + "/open_lth/")
        print(f"Appending paths: {str(parent_dir_path)}")
        
        #pretraining step:

        #get config files from parser
        self.reader = parser.readConfig(config_file)

        #get the number of models to train
        self.num_models = int(self.reader.exp_info["number of models"])
        self.num_jobs = int(self.reader.exp_info["number of jobs"])

        #number of models to train per job
        if self.num_models % self.num_jobs == 0:
            self.num_train_per_job = numpy.full(self.num_jobs, self.num_models/self.num_jobs).astype(int)
        else:
            self.num_train_per_job = numpy.full(self.num_jobs - 1, int(self.num_models/self.num_jobs)).astype(int)
            self.num_train_per_job = numpy.append(self.num_train_per_job, int(self.num_models % self.num_jobs)) #check this

        #make output directories
        self.reader.mk_directories(self.num_train_per_job)

        """
        TRAINING STEP
        """
        
        print(f"Division of jobs (models/job): {self.num_train_per_job}")
        print(f"Starting training...")
        #training step:
        #and for each job, pass models onto trainer
        job_idx = 0
        model_idx = 0
        for job in self.reader.jobs:
            print(f"{job}: {self.reader.jobs[job]}")
            for model_no in range(self.num_train_per_job[job_idx]):
                model = self.reader.get_model(job)
                model_trainer = trainer.train(model, self.reader.exp_info, self.reader.jobs[job], job_idx, model_idx)
                model_trainer.trainLoop(model)
                model_idx+=1
            model_idx=0
            job_idx+=1
        
        """
        PROCESS STEP
        """
        print(f"Now processing output...")
        crtForget = createforgetdataset.createForgetDataset()

        dmg = damagemodel.damageModel()