import numpy as np
import torch
from torch.utils.data import Subset
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from aff_nist import affNIST


def get_train_valid_loader(data_dir,
                           dataset,
                           batch_size,
                           num_workers=4,
                           pin_memory=False):
    data_dir = data_dir + '/' + dataset

    if dataset == "cifar10":
        train_trans = [transforms.RandomResizedCrop(32, scale=(0.75, 1.0), ratio=(1.0, 1.0)),
                       transforms.Resize((8, 8), interpolation=InterpolationMode.BICUBIC),
                       transforms.Resize((64, 64), interpolation=InterpolationMode.BICUBIC),
                       transforms.RandomHorizontalFlip(p=0.5),
                       transforms.RandAugment(num_ops=1, magnitude=8),
                       transforms.ColorJitter(0.1, 0.1, 0.1),
                       transforms.ToTensor(),
                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                       transforms.RandomErasing(p=0.25)]

        # Define transforms for the test/validation set
        test_trans = [transforms.Resize((8, 8), interpolation=InterpolationMode.BICUBIC),
                      transforms.Resize((64, 64), interpolation=InterpolationMode.BICUBIC),
                      transforms.ToTensor(),
                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]

        # Load the training set
        train_set = datasets.CIFAR10(data_dir, train=True, download=True,
                                     transform=transforms.Compose(train_trans))

        # Load the test set (to be used as validation set)
        valid_set = datasets.CIFAR10(data_dir, train=False, download=True,
                                     transform=transforms.Compose(test_trans))

        # Data loaders
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        valid_loader = torch.utils.data.DataLoader(
            valid_set, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

    elif dataset == "mnist":
        train_trans = [transforms.RandomCrop(32, padding=4),
                       transforms.Resize((64, 64), interpolation=InterpolationMode.BICUBIC),
                       transforms.RandomHorizontalFlip(0.5),
                       transforms.RandomAffine(degrees=0, translate=(0.3, 0.3),
                                               interpolation=InterpolationMode.BICUBIC),
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))]

        test_trans = [transforms.Resize((64, 64), interpolation=InterpolationMode.BICUBIC),
                      transforms.ToTensor(),
                      transforms.Normalize((0.1307,), (0.3081,))]

        train_set = datasets.MNIST(data_dir, train=True, download=True, transform=transforms.Compose(train_trans))
        valid_set = datasets.MNIST(data_dir, train=False, download=True, transform=transforms.Compose(test_trans))
        # valid_set = affNIST('./data/Affnist', is_Train=False, transform=transforms.Compose(test_trans))

        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        valid_loader = torch.utils.data.DataLoader(
            valid_set, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

    elif dataset == "svhn":
        # Define transforms for the training set
        train_trans = [transforms.RandomResizedCrop(32, scale=(0.75, 1.0), ratio=(1.0, 1.0),
                                                    interpolation=InterpolationMode.BICUBIC),
                       transforms.Resize((8, 8), interpolation=InterpolationMode.BICUBIC),
                       transforms.Resize((64, 64), interpolation=InterpolationMode.BICUBIC),
                       transforms.RandomHorizontalFlip(p=0.5),
                       transforms.RandAugment(num_ops=1, magnitude=8),
                       transforms.ColorJitter(0.1, 0.1, 0.1),
                       transforms.ToTensor(),
                       transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614)),
                       transforms.RandomErasing(p=0.25)]

        # Define transforms for the test/validation set
        test_trans = [transforms.Resize((8, 8), interpolation=InterpolationMode.BICUBIC),
                      transforms.Resize((64, 64), interpolation=InterpolationMode.BICUBIC),
                      transforms.ToTensor(),
                      transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))]

        # Load the training set
        train_set = datasets.SVHN(data_dir, split='train', download=True, transform=transforms.Compose(train_trans))

        extra_set = datasets.SVHN(data_dir, split='extra', download=True, transform=transforms.Compose(train_trans))
        extra_limit = 250000  # Choose only 250k extra samples
        extra_set.data = extra_set.data[:extra_limit]
        extra_set.labels = extra_set.labels[:extra_limit]
        data = np.concatenate([train_set.data, extra_set.data], axis=0)
        labels = np.concatenate([train_set.labels, extra_set.labels], axis=0)
        train_set.data = data
        train_set.labels = labels

        # Load the test set (to be used as validation set
        valid_set = datasets.SVHN(data_dir, split='test', download=True, transform=transforms.Compose(test_trans))

        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_set, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

    elif dataset == "cifar100":
        train_trans = [transforms.RandomResizedCrop(32, scale=(0.75, 1.0), ratio=(1.0, 1.0),
                                                    interpolation=InterpolationMode.BICUBIC),
                       transforms.Resize((8, 8), interpolation=InterpolationMode.BICUBIC),
                       transforms.Resize((64, 64), interpolation=InterpolationMode.BICUBIC),
                       transforms.RandomHorizontalFlip(p=0.5),
                       transforms.RandAugment(num_ops=1, magnitude=8),
                       transforms.ColorJitter(0.1, 0.1, 0.1),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5074, 0.4867, 0.4411), (0.2011, 0.1987, 0.2025)),
                       transforms.RandomErasing(p=0.25)]

        # Define transforms for the test/validation set
        test_trans = [transforms.Resize((64, 64), interpolation=InterpolationMode.BICUBIC),
                      transforms.ToTensor(),
                      transforms.Normalize((0.5074, 0.4867, 0.4411), (0.2011, 0.1987, 0.2025))]

        # Load the training set
        train_set = datasets.CIFAR100(data_dir, train=True, download=True,
                                      transform=transforms.Compose(train_trans))

        # Load the test set (to be used as validation set)
        valid_set = datasets.CIFAR100(data_dir, train=False, download=True,
                                      transform=transforms.Compose(test_trans))

        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        valid_loader = torch.utils.data.DataLoader(
            valid_set, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

    else:
        print("Unsupported dataset!")

    return train_loader, valid_loader


def get_test_loader(data_dir,
                    dataset,
                    batch_size,
                    num_workers=4,
                    pin_memory=False):
    data_dir = data_dir + '/' + dataset

    if dataset == "cifar10":
        trans = [transforms.Resize((8, 8), interpolation=InterpolationMode.BICUBIC),
                 transforms.Resize((64, 64), interpolation=InterpolationMode.BICUBIC),
                 transforms.ToTensor(),
                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
        dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                   transform=transforms.Compose(trans))

    elif dataset == "svhn":
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [109.9, 109.7, 113.8]],
                                         std=[x / 255.0 for x in [50.1, 50.6, 50.8]])
        trans = [transforms.ToTensor(),
                 normalize]
        dataset = datasets.SVHN(data_dir, split='test', download=True,
                                transform=transforms.Compose(trans))

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader


DATASET_CONFIGS = {
    'cifar10': {'size': 64, 'channels': 3, 'classes': 10},
    'svhn': {'size': 64, 'channels': 3, 'classes': 10},
    'cifar100': {'size': 64, 'channels': 3, 'classes': 100},
    'mnist': {'size': 64, 'channels': 1, 'classes': 10},
}
