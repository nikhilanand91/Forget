from dataclasses import dataclass, field
from torchvision import datasets, transforms
from typing import List
from torch.utils.data import DataLoader
from base.dataset import Dataset

@dataclass
class Cifar10(Dataset):

    train: bool = True
    transforms_to_apply = None
    download: bool = True

    def __post_init__(self):
        self.dataset = None

    def get_dataset(self, save_location: str = '/'):           
        self.dataset = datasets.CIFAR10(root = save_location, 
                                train = self.train, 
                                download = self.download, 
                                transform = self.transforms_to_apply)

        return self.dataset

    def get_dataloader(self, batch_size: int, shuffle: bool = False):
        if self.dataset != None:
            return DataLoader(dataset = self.dataset, batch_size = batch_size, shuffle = shuffle)
        else:
            raise ValueError('No dataset found!')
            sys.exit(1)

    def get_relative_ordering(self, batch, dataloader: DataLoader):
        """Get relative ordering of examples in a batch to the loaded dataset."""
        if self.dataset == None:
            raise ValueError('Set the dataset and dataloader first!')
            sys.exit(1)

        order = []
        for example, labels in enumerate(batch):
            for batch_idx, original_batch in enumerate(dataloader):
                orig_ex, orig_labels = original_batch
                for ex_idx, ex in enumerate(orig_ex):
                    if all(torch.eq(example, ex)):
                        order.append([batch_idx, ex_idx])
        return order

    def get_absolute_ordering(self, batch, dataloader: DataLoader):
        if self.dataset == None:
            raise ValueError('Set the dataset and dataloader first!')
            sys.exit(1)

        

    def set_transforms(self, transform = transforms.Compose([transforms.ToTensor(), \
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])):
        if transform != None:
            self.transforms_to_apply = transform