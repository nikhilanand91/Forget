from datasets.cifar10 import Cifar10

allowed_datasets = {'cifar10': Cifar10()}

def get_dataset_object(dataset_name: str, save_location: str, shuffle: bool = False):
    if dataset_name not in allowed_datasets:
        raise ValueError(f'Dataset {dataset_name} not found!')
    else:
        dataset = allowed_datasets[dataset_name]
        dataset.save_location = save_location
        dataset.shuffle = shuffle

    dataset.set_transforms()
    return dataset