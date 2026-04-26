import torch
import torch.nn as nn
def train_one_epoch(model, loader, optimizer, criterion, device):
  model.train()
  loss=0
  accuracy=0
  samples=0
  for input, labels in loader:
    input=input.to(device)
    labels=labels.to(device)
    optimizer.zero_grad()
    output=model(input)
    preds=torch.argmax(output, dim=1)
    loss_value=criterion(output, labels)
    loss_value.backward()
    loss+=loss_value.item()
    optimizer.step()
    samples+=input.size(0)
    accuracy+=(preds==labels).sum().item()
  loss=loss/len(loader)
  accuracy=accuracy/samples
  return loss, accuracy
def evaluate(model, loader, criterion, device):
  model.eval()
  with torch.no_grad():
    loss=0
    accuracy=0
    samples=0
    for input, labels in loader:
      input=input.to(device)
      labels=labels.to(device)
      output=model(input)
      preds=torch.argmax(output, dim=1)
      loss_value=criterion(output, labels)
      loss+=loss_value.item()
      samples+=input.size(0)
      accuracy+=(preds==labels).sum().item()
    loss=loss/len(loader)
    accuracy=accuracy/samples
    return loss, accuracy
def train(model, train_loader, val_loader, optimizer, scheduler, criterion, device, epochs):
    history_dict = {
        "train_loss": [],
        "train_acc" : [],
        "val_loss"  : [],
        "val_acc"   : [],
    }
    best_val_acc = 0.0

    for i in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        eval_loss,  eval_acc  = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history_dict["train_loss"].append(train_loss)
        history_dict["train_acc"].append(train_acc)
        history_dict["val_loss"].append(eval_loss)
        history_dict["val_acc"].append(eval_acc)

        if eval_acc > best_val_acc:
            best_val_acc = eval_acc
            torch.save({
                "epoch"    : i,
                "model"    : model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_acc"  : best_val_acc,
            }, "nest_best.pt")

        print(f"Epoch {i+1:3d}/{epochs} | "
              f"train loss {train_loss:.4f} acc {train_acc*100:.2f}% | "
              f"val loss {eval_loss:.4f} acc {eval_acc*100:.2f}% | "
              f"best {best_val_acc*100:.2f}%")

    return history_dict
