from dataclasses import dataclass, field
from torchvision import datasets, transforms
from typing import List

@dataclass
class Dataset:

    dataset_name: str
    output_location: str

    train: bool = True
    transforms_to_apply = None
    download: bool = False
    num_workers: int = 0
    allowed_datasets: List[str] = field(default_factory = lambda: ['cifar10'])

    def get_dataset():
        if self.dataset_name not in self.allowed_datasets:
            raise ValueError(f'Dataset not found! Make sure it is one of: {self.allowed_datasets}')

        if self.dataset_name == 'cifar10':
            transform = self.transforms_to_apply if self.transforms_to_apply != None else \
                        transforms.Compose([self.transforms_to_apply.ToTensor(), 
                                            self.transforms_to_apply.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

            return datasets.CIFAR10(root = self.output_location, 
                                    train = self.train, 
                                    download = self.download, 
                                    transform = transform,
                                    num_workers = self.num_workers)