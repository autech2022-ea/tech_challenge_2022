import numpy as np
import torchvision
import transforms as tr
from torch.utils.data import SubsetRandomSampler


DEFAULT_DATASET_LOCATION = './data'


def class_to_id():
    dummy_dataset = torchvision.datasets.ImageFolder(DEFAULT_DATASET_LOCATION)
    return dummy_dataset.find_classes(DATA_DIRECTORY)[1]


def id_to_class():
    dummy_dataset = torchvision.datasets.ImageFolder(DEFAULT_DATASET_LOCATION)
    class_to_id = dummy_dataset.find_classes(DEFAULT_DATASET_LOCATION)[1]
    return dict((v, k) for k, v in class_to_id.items())


def get_datasets(location, tile_size=256, val_ratio=0.15, seed=1987):
    """
    Get a couple of tuples. Each tuple is
    (dataset, SubsetRandomSampler)
    Normally each of the two datasets would return all the elements, that
    is why the SubsetRandomSampler are necessary so they return
    train OR val indexes only

    :param str location: location of the root of the dataset directory
    :param int tile_size: images will be smartly resized to tile_size x tile_size
    :param float val_ratio: percentage used as validation (from 0 to 1)
    :param int seed: seed to obtain deterministic results on the train/val split
    :return: (train_dataset, train_sampler), (val_dataset, val_sampler)
    """
    train_dataset = torchvision.datasets.ImageFolder(
        location,
        transform=tr.get_train_compose(tile_size)
    )
    val_dataset = torchvision.datasets.ImageFolder(
        location,
        transform=tr.get_val_compose(tile_size)
    )

    total_instances = len(train_dataset)
    train_len = int(total_instances * (1 - val_ratio))
    val_len = total_instances - train_len

    indices = list(range(total_instances))
    np.random.seed(seed)  # make it deterministic
    np.random.shuffle(indices)

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(indices[val_len:])
    val_sampler = SubsetRandomSampler(indices[:val_len])

    return (train_dataset, train_sampler), (val_dataset, val_sampler)
