import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from train.callbacks import History, Metrics, def_metrics
import numpy as np

class EarlyStopping:
    def __init__(self, patience=10, delta=0, path='checkpoint.pt', verbose=True):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model, save_best_acc = False):
        self.save_best_acc = save_best_acc
        if save_best_acc:
            score = val_loss
        else:
            score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        if self.verbose:
            if self.save_best_acc:
                print(f'Validation accuracy increased ({0 if self.val_loss_min == np.Inf else self.val_loss_min * 100:.6f} --> {val_loss*100:.6f}).  Saving model ...')
            else:    
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

class TrainElements:
    def __init__(self, device, criterion = None, optimizer = None, history = None, metrics = None, earlystopping = None):
        self.criterion = criterion
        self.optimizer = optimizer
        self.history = history
        self.metrics = metrics
        self.earlystopping = earlystopping
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        if optimizer is None:
            self.optimizer = optim.Adam
        if history is None:
            self.history = History()
        if metrics is None:
            self.metrics = def_metrics(device)
        if earlystopping is None:
            self.earlystopping = EarlyStopping()

def get_model_size(model):
    return sum(p.numel() for p in model.parameters())




