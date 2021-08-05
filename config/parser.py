from dataclasses import dataclass

@dataclass
class readConfig:
    config_file: str = "default_config.ini"
    config = config.read(config_file)

    def __post_init__(self):
        self.sections = config.sections()
        self.jobs = {}

        for section in self.sections:
            if section == "Experiment info":
                options = config.options(section)
                self.experiment_name = config.get(section, options[0])
                self.experiment_directory = config.get(section, options[1])
                self.num_models_train = int(config.get(section, options[2]))
                self.num_jobs = int(config.get(section, options[3]))
            elif str.split(section)[0] == "Job" and str.split(section)[1].isdigit():
                self.jobs[section] = {}
                options = config.options(section)
                #change this to a loop
                for i in range(len(options)):
                    self.jobs[section][str(options[i])] = config.get(section, options[i])
            else:
                raise ValueError("Unknown section command in config file!")