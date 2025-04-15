from tqdm import tqdm
import torch
from utils import *

def training_step(train_loader, model, optimizer, device, criterion):
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, desc="Training", leave=True, dynamic_ncols=True)

    for batch, (X, y) in enumerate(loop):
        # Send data to GPU
        X, y = X.to(device), y.to(device)

        # Forward Pass
        y_pred = model(X)

        # Calculate loss
        loss_1 = criterion(y_pred, y)
        total_loss += loss_1.item()

        # Optimizer reset step
        optimizer.zero_grad()

        # Loss Backpropagation
        loss_1.backward()

        # Optimizer step
        optimizer.step()

    total_loss /= len(train_loader)
    return total_loss


def validation_step(val_loader, model, device, criterion):
    model.eval()
    total_loss = 0
    loop = tqdm(val_loader, desc="Validation", leave=True)

    with torch.inference_mode():
        for batch, (X, y) in enumerate(loop):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)

            # Calculate structural similarity index loss
            loss_1 = criterion(y_pred, y)
            total_loss += loss_1.item()

    total_loss /= len(val_loader)

    return total_loss


def train(train_loader,
          val_loader,
          model,
          epochs,
          optimizer,
          device,
          best_model_path,
          criterion,
          initial_lr,
          patience=20):
    early_stopping = EarlyStopping(patience=patience, mode='min')

    for epoch in tqdm(range(epochs), desc="All"):
        # Adjust learning rate
        current_lr = adjust_learning_rate(optimizer, epoch, initial_lr)
        print(f"Epoch: {epoch}, Learning Rate: {current_lr:.6f}\n-----------")
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        train_loss = training_step(train_loader, model, optimizer, device, criterion)
        val_loss = validation_step(val_loader, model, device, criterion)

        print(
            f"Train loss: {train_loss:.8f} | "
            f"Val loss: {val_loss:.8f} | "
        )
        print("-------------\n")

        if check_early_stopping(val_loss, model, early_stopping, epoch, best_model_path):
            break

    # Ripristina i pesi migliori
    model.load_state_dict(torch.load(best_model_path))
    print(f"Restored best model weights with val_loss: {early_stopping.best_val:.4f}")