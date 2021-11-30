from datasets.cifar10 import Cifar10

allowed_datasets = {'cifar10': Cifar10()}

def get_dataset(dataset_name: str, save_location: str):
    if dataset_name not in allowed_datasets:
        raise ValueError(f'Dataset {dataset_name} not found!')
    else:
        dataset = allowed_datasets[dataset_name]

    dataset.set_transforms()
    return dataset.get_dataset(save_location = save_location)