from dataclasses import dataclass, field
from torchvision import datasets, transforms
from typing import List
from torch.utils.data import DataLoader, SequentialSampler
from base.dataset import Dataset
from base.random_sampler import RandomSampler

@dataclass
class Cifar10(Dataset):

    train: bool = True
    transforms_to_apply = None
    download: bool = True
    shuffle: bool = False

    def __post_init__(self):
        self.dataset = None
        self.ordering = None
        self._default_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.sampler = None
        self.save_location = '/'

    def get_dataset(self):           
        self.dataset = datasets.CIFAR10(root = self.save_location, 
                                train = self.train, 
                                download = self.download, 
                                transform = self.transforms_to_apply)

        return self.dataset

    def get_dataloader(self, batch_size: int):
        if not self.dataset or not self.sampler:
            raise ValueError('No dataset or sampler found!')
            sys.exit(1)

        return DataLoader(dataset = self.dataset, batch_size = batch_size, sampler = self.sampler)

    def set_transforms(self, transform = None):
        transform = self._default_transform if not transform else _
        self.transforms_to_apply = transform

    def get_sampler(self):
        return RandomSampler() if self.shuffle else SequentialSampler(self.dataset)

    def get_ordering(self):
        if not self.sampler:
            raise ValueError('No sampler found!')
            sys.exit(1)

        if shuffle:
            return self.sampler.shuffle()
        else:
            sequence = range(len(self.dataset))
            return iter([sequence[i:i + batch_size] for i in range(0, len(sequence), batch_size)])