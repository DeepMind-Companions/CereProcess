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


class EEGDatasetS(Dataset):
    def __init__(self, files, labels):
        """
        Args:
            annotations_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.files = files
        self.labels = labels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        eeg_name = self.files[idx]
        eeg = np.load(eeg_name, allow_pickle=True)['data']
        eeg = torch.tensor(eeg)
        label = self.labels[idx]
        # One hot encode the label
        if label == 0:
            label = torch.tensor([1, 0])
        else:
            label = torch.tensor([0, 1])
        return eeg, label

def get_datasets(train_dir, val_dir, shuffle = 0):
    if (shuffle == 0):
        train_dataset = EEGDataset(train_dir)
        val_dataset = EEGDataset(val_dir)
    else:


        # Gathering the annotations and files in advance

        # Setting the seed according to shuffle value
        np.random.seed(shuffle)

        # Getting all annotations
        annotations = os.path.join(train_dir, 'data.csv')
        annotations_train = pd.read_csv(annotations)
        annotations = os.path.join(val_dir, 'data.csv')
        annotations_val = pd.read_csv(annotations)
        

        # Getting all files and labels
        train_files = [os.path.join(train_dir, file) for file in annotations_train.iloc[:, 0]]
        train_labels = [label for label in annotations_train.iloc[:, 1]]
        val_files = [os.path.join(val_dir, file) for file in annotations_val.iloc[:, 0]]
        val_labels = [label for label in annotations_val.iloc[:, 1]]
        # Counting the number of eval abnormal files
        abnormal_count = 0
        for label in val_labels:
            if label == 1:
                abnormal_count += 1

        val_len = len(val_files)
        normal_count = val_len - abnormal_count

        # add train and val together
        train_files.extend(val_files)
        train_labels.extend(val_labels)

        # Shuffle train and val together according to same ratio
        idx = np.random.permutation(len(train_files))
        train_files = [train_files[i] for i in idx]
        train_labels = [train_labels[i] for i in idx]

        # Split train and val with same ratio of abnormal files
        val_files = []
        val_labels = []
        tr_files = []
        tr_labels = []
        for i in range(len(train_labels)):
            if train_labels[i] == 1:
                if (abnormal_count > 0):
                    val_files.append(train_files[i])
                    val_labels.append(train_labels[i])
                    abnormal_count -= 1
                else:
                    tr_files.append(train_files[i])
                    tr_labels.append(train_labels[i])
            else:
                if (normal_count <= 0):
                    tr_files.append(train_files[i])
                    tr_labels.append(train_labels[i])
                else:
                    val_files.append(train_files[i])
                    val_labels.append(train_labels[i])
                    normal_count -= 1


        
        # Train and val dataset separately now
        train_dataset = EEGDatasetS(tr_files, tr_labels)
        val_dataset = EEGDatasetS(val_files, val_labels)

    return train_dataset, val_dataset



