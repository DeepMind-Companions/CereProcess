import os
import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd


class EEGDataset(Dataset):
    def __init__(self, root_dir):
        """
        Args:
            annotations_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        annotations_file = os.path.join(root_dir, 'data.csv')
        self.annotations = pd.read_csv(annotations_file)
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


