from dataclasses import dataclass, field
from torchvision import datasets, transforms
from typing import List
from torch.utils.data import DataLoader

@dataclass
class Dataset:

    dataset_name: str
    output_location: str

    train: bool = True
    transforms_to_apply = None
    download: bool = True
    allowed_datasets: List[str] = field(default_factory = lambda: ['cifar10'])

    def get_dataset(self):
        if self.dataset_name not in self.allowed_datasets:
            raise ValueError(f'Dataset not found! Make sure it is one of: {self.allowed_datasets}')

        if self.dataset_name == 'cifar10':
            transform = self.transforms_to_apply if self.transforms_to_apply != None else \
                        transforms.Compose([transforms.ToTensor(), 
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

            return datasets.CIFAR10(root = self.output_location, 
                                    train = self.train, 
                                    download = self.download, 
                                    transform = transform)

    def get_relative_ordering(self, batch, dataloader: DataLoader):
        """Get relative ordering of examples in a batch to a dataloader.
        Useful utility function when dataloader is shuffled between epochs."""
        order = []
        for example, labels in enumerate(batch):
            for batch_idx, original_batch in enumerate(dataloader):
                orig_ex, orig_labels = original_batch
                for ex_idx, ex in enumerate(orig_ex):
                    if all(torch.eq(example, ex)):
                        order.append([batch_idx, ex_idx])
        return order