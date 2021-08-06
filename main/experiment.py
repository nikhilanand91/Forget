class experiment:
    #the experiment should call on config.py to get info and create the appropriate directories
    #based on the contents of config.ini file. It should then be divided into three steps:
    #1. Pretraining (e.g. load model from OpenLTH)
    #2. Training (for each job, pass models onto trainer.py which trains it and stores the data)
    #3. Post-training (do stuff with the data - e.g., from postprocess.py and plotter.py plot statistics of forgetting events)

    #pretraining step: