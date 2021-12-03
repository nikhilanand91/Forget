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

    def __post_init__(self):
        self.dataset = None
        self.ordering = None
        self._default_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def get_dataset(self, save_location: str = '/'):           
        self.dataset = datasets.CIFAR10(root = save_location, 
                                train = self.train, 
                                download = self.download, 
                                transform = self.transforms_to_apply)

        return self.dataset

    def get_dataloader(self, batch_size: int, shuffle: bool = False):
        if self.dataset == None:
            raise ValueError('No dataset found!')
            sys.exit(1)

        if shuffle:
            sampler = RandomSampler()
            self.ordering = sampler.shuffle()
        else:
            sampler = SequentialSampler(self.dataset)
            sequence = range(len(self.dataset))
            self.ordering = iter([sequence[i:i + batch_size] for i in range(0, len(sequence), batch_size)])

        return DataLoader(dataset = self.dataset, batch_size = batch_size, sampler = sampler)

    def set_transforms(self, transform = None):
        transform = self._default_transform if not transform else _
        self.transforms_to_apply = transform