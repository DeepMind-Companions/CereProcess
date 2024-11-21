import os
import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import StratifiedKFold

class EEGDataset(Dataset):
    def __init__(self, root_dir, indexes=None):
        """
        Args:
            annotations_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the data.
        """
        annotations_file = os.path.join(root_dir, 'data.csv')
        self.annotations = pd.read_csv(annotations_file)
        if (type(indexes) != type(None)):
            self.annotations = self.annotations[self.annotations.index.isin(indexes)]
        self.root_dir = root_dir

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        eeg_name = os.path.join(self.root_dir,
                                self.annotations.iloc[idx, 0])
        eeg = np.load(eeg_name, allow_pickle=True)['data']
        eeg = torch.tensor(eeg)
        label = self.annotations.iloc[idx, 1]
        # One hot encode the label
        if label == 0:
            label = torch.tensor([1, 0])
        else:
            label = torch.tensor([0, 1])
        return eeg, label


class KFoldDataset():
    def __init__(self, root_dir, n_splits=10, shuffle=True, random_state=42):
        """
        Args:
            annotations_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the data.
        """
        annotations_file = os.path.join(root_dir, 'data.csv')
        self.root_dir = root_dir
        self.annotations = pd.read_csv(annotations_file)
        self.skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        self.gen = self.skf.split(self.annotations["File"], self.annotations["Label"])

    def __iter__(self):
        return self

    def __next__(self):
        curr = next(self.gen)
        return EEGDataset(self.root_dir, curr[0]), EEGDataset(self.root_dir, curr[1])
