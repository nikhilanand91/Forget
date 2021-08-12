"""
Class for managing forget data. It can go through forget statistics and output
a mask corresponding to the forget threshold. This mask can be used e.g. to obtain a
dataset of forgettable examples. It can also output subset datasets of correctly
classified examples and forgotten + correctly classified examples at the end of
training.
"""

import torch
import os
import sys
sys.path.append(str("/")) #this will need to be fixed later
from Forget.config import parser
import os

class createForgetDataset:
    def __init__(self, forget_thres = 3, config_file = "/Forget/config/default_config.ini"):
        #assume latest epoch, go through forgetstats and correctstats
        #for each model, initialize the below variables, create the forget
        #dataset + save into forgetdata/ and also the forget+correct dataset.

        #first set the threshold
        self.forget_thres = forget_thres

        #from the config file, obtain experiment name + number of jobs + number of models
        self.reader = parser.readConfig(config_file)
        

        self.exp_name = self.reader.exp_info["name"]
        #self.num_jobs = int(self.reader.exp_info["number of jobs"])
        
        print(f"Reading from output files in experiment {self.exp_name}...")
        for job in self.reader.jobs: #remember to put everything into forgetdata
            self.list_forget_folders = [self.exp_name + "/" + job + "/" + f.name + "/forgetdata/" for f in os.scandir("/" + self.exp_name + "/" + job + "/") if f.is_dir()]
            #self.list_correct_folders = [self.exp_name + "/" + job + "/" + f.name + "/correctdata/" for f in os.scandir("/" + self.exp_name + "/" + job + "/") if f.is_dir()]
            self.max_epoch = int(self.reader.jobs[job]["num epochs"])
        
        """
        for each model go through forgetstats and correctstats, upload the masks to the folder.
        This part is undoubtedly clunky... ideally we'd have a directory just associated with the model
        and scan over models, rather than scan over directories.
        Maybe something to improve in the next iteration of this code.
        """

        for subfolder in self.list_forget_folders:
            self.forget_stats = torch.load("/" + subfolder + "forgetstatsepoch="+str(self.max_epoch)+".pt")
            self.correct_stats = torch.load("/" + subfolder + "correctstatsepoch="+str(self.max_epoch)+".pt")
            self.sum_over_ep_flatten_forget = torch.flatten(torch.sum(self.forget_stats, 0))

        print(self.forget_statistics)
            #for path in list_subfolders_with_paths: