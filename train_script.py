import argparse
import torch
import torch.nn as nn
from configs.cifar10 import get_config as get_cifar10_config
from configs.cifar100 import get_config as get_cifar100_config
from models.nest import NeST
from data.dataset import get_dataloaders
from engine.train import train

def get_args():
    parser = argparse.ArgumentParser(description="Train NeST")
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--dataset",    type=str,   default="cifar10",
                        choices=["cifar10", "cifar100"])
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--device",     type=str,   default="cuda")
    return parser.parse_args()

def main():
    args = get_args()

    if args.dataset == "cifar10":
        config = get_cifar10_config()
    else:
        config = get_cifar100_config()

    config["device"] = torch.device(args.device
                                    if torch.cuda.is_available()
                                    else "cpu")

    train_loader, val_loader = get_dataloaders(config)

    model     = NeST(config).to(config["device"])
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    print(f"Dataset  : {args.dataset}")
    print(f"Epochs   : {args.epochs}")
    print(f"Device   : {config['device']}")
    print(f"Params   : {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    train(model, train_loader, val_loader,
          optimizer, scheduler, criterion,
          config["device"], args.epochs)

if __name__ == "__main__":
    main()