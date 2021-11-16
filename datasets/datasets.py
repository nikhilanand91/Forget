from dataclasses import dataclass
from torchvision import datasets, transforms
from typing import List

@dataclass
class Dataset:

    dataset_name: str
    output_location: str

    train: bool = True
    transforms: None
    download: bool = False
    num_workers: int = 0
    allowed_datasets: List = ['cifar10']

    def get_dataset():
        if self.dataset_name not in self.allowed_datasets:
            raise ValueError(f'Dataset not found! Make sure it is one of: {self.allowed_datasets}')

        if self.dataset_name == 'cifar10':
            transform = transforms if transforms != None else \
                        transforms.Compose([transforms.ToTensor(), 
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

            return datasets.CIFAR10(root = self.output_location, 
                                    train = self.train, 
                                    download = self.download, 
                                    transform = transform,
                                    num_workers = self.num_workers)