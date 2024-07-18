import torch
from tqdm.notebook import tqdm
from IPython.display import clear_output


def eval(model, val_loader, criterion, device, metrics, history):
    val_loss = 0
    model.eval()
    metrics.reset()
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()
            _, predicted = torch.max(output, 1)
            label_check = torch.argmax(target, 1)
            metrics.update(label_check, predicted)
        val_loss /= len(val_loader)
        history.update(metrics.compute().update({"loss": val_loss}), 'val')
    return val_loss

def train(model, train_loader, val_loader, optimizer, criterion, epochs, history, metrics, device, save_path, scheduler=None):
    model.to(device)
    best_val_loss = float('inf')
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        metrics.reset()
        for data, target in tqdm(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(output, 1)
            label_check = torch.argmax(target, 1)
            train_loss += loss.item()
            metrics.update(label_check, predicted)
        train_loss /= len(train_loader)
        history.update(metrics.compute().update({"loss": train_loss}), 'train');
        val_loss = eval(model, val_loader, criterion, device, metrics, history)
        model.train()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
        if scheduler:
            scheduler.step(val_loss)
        clear_output(wait=True)
        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}', flush=True)
    model.load_state_dict(torch.load(save_path))
    return model
