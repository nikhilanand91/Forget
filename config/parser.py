import configparser
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
                self.jobs[section]["model parameters"] = config.get(section, options[0])
                self.jobs[section]["save models"] = config.get(section, options[1])
                self.jobs[section]["num epochs"] = int(config.get(section, options[2]))
                self.jobs[section]["dataset"] = config.get(section, options[3])
                self.jobs[section]["measure forget"] = config.get(section, options[4])
                self.jobs[section]["track correct examples"] = config.get(section, options[5])
                self.jobs[section]["save forget dataset"] = config.get(section, options[6])
                self.jobs[section]["job directory"] = config.get(section, options[7])
                self.jobs[section]["add model noise"] = config.get(section, options[8])
                self.jobs[section]["noise parameters"] = config.get(section, options[9])
                self.jobs[section]["save clones"] = config.get(section, options[10])
            else:
                raise ValueError("Unknown section command in config file!")