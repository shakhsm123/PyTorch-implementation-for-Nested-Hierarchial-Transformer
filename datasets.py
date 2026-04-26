import torch
import torchvision
import torchvision.transforms as T

def get_dataloaders(config):
    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.4914, 0.4822, 0.4465],
                    std =[0.2023, 0.1994, 0.2010]),
    ])
    val_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.4914, 0.4822, 0.4465],
                    std =[0.2023, 0.1994, 0.2010]),
    ])

    if config["num_classes"] == 100:
        dataset_cls = torchvision.datasets.CIFAR100
    else:
        dataset_cls = torchvision.datasets.CIFAR10

    train_dataset = dataset_cls(root="./data", train=True,
                                download=True, transform=train_transform)
    val_dataset   = dataset_cls(root="./data", train=False,
                                download=True, transform=val_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128,
                                               shuffle=True,  num_workers=2,
                                               pin_memory=True)
    val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=128,
                                               shuffle=False, num_workers=2,
                                               pin_memory=True)
    return train_loader, val_loader