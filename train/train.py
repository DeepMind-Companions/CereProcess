import torch
import torch.nn.functional as F
from tqdm.notebook import tqdm
from IPython.display import clear_output


def evaluate(model, val_loader, criterion, device, metrics, history):
    model.to(device)
    val_loss = 0
    model.eval()
    metrics.reset()
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            target = target.float()
            data = data.float()
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()
            _, predicted = torch.max(output, 1)
            label_check = torch.argmax(target, 1)
            metrics.update(label_check, predicted)
        val_loss /= len(val_loader)
        results = metrics.compute()
        results.update({"loss": val_loss})
        history.update(results, 'val')
    return val_loss

def train(model, train_loader, val_loader, optimizer, criterion, epochs, history, metrics, device, save_path, earlystopping, scheduler=None):
    model = model.to(device)
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        metrics.reset()
        for data, target in tqdm(train_loader):
            data, target = data.to(device), target.to(device)
            data = data.float()
            target = target.float()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            output = F.softmax(output, dim = -1)
            _, predicted = torch.max(output, 1)
            label_check = torch.argmax(target, 1)
            train_loss += loss.item()

            # clearing data for space
            del data, target, output, loss
            if device == 'cuda':
                torch.cuda.empty_cache()

            metrics.update(label_check, predicted)
        train_loss /= len(train_loader)
        results = metrics.compute()
        results.update({"loss": train_loss})
        history.update(results, 'train')
        val_loss = evaluate(model, val_loader, criterion, device, metrics, history)
        model.train()
        clear_output(wait=True)
        earlystopping(val_loss, model)
        if scheduler:
            scheduler.step(val_loss)
        if earlystopping.early_stop:
            print("Early stopping")
            break
        if device == 'cuda':
            torch.cuda.empty_cache()
        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}', flush=True)
        print(f'Train Accuracy: {float(history.history["train"]["accuracy"][-1]):.4f} - Val Accuracy: {float(history.history["val"]["accuracy"][-1]):.4f}', flush=True)
        print(f'Train F1 Score: {float(history.history["train"]["f1score"][-1]):.4f} - Val F1 Score: {float(history.history["val"]["f1score"][-1]):.4f}', flush = True)
    model.load_state_dict(torch.load(save_path))
    return model
