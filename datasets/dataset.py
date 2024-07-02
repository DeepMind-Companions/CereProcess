"""dataset includes the classes and functions necessary to load and preprocess the dataset.

"""
import mne
import csv
import numpy as np
from tqdm import tqdm
import random
import os
from .getfiles import get_files, get_traineval, get_filedir
from .pipeline import Pipeline

class Dataset:
    ''' Dataset class stores the EEG data, defining a preprocessing pipeline and converting it to other forms accordingly.
        Keeps Data in EEG format and then converts it to other forms

    '''
    def __init__(self, datapath, basedir = 'TUH EEG Corpus/edf'):
        ''' Constructor Function
            INPUT:
                datapath - string - path to the MNE source files
                basedir - string - the directory before the train and eval directories
        '''
        self.traindir, self.evaldir = get_traineval(datapath, basedir)
        self.pipeline = Pipeline()

    def set_pipeline(self, pipeline):
        ''' Adds a pipeline to the dataset
            INPUT:
                pipeline - Pipeline - the pipeline to add
        '''
        self.pipeline = pipeline

    def add_pipeline(self, pipeline):
        ''' Adds a pipeline to the dataset
            INPUT:
                pipeline - Pipeline - the pipeline to add
        '''
        self.pipeline = self.pipeline + pipeline

    def load_data(self, div = 'train', base_dir = '01_tcp_ar'):
        ''' Loads the data from the dataset
            INPUT:
                div - string - the division of the dataset to load
            OUTPUT:
                data - EEG - the data loaded from the dataset
        '''
        if div == 'train':
            dir = self.traindir
        elif div == 'eval':
            dir = self.evaldir
        else:
            raise ValueError('Invalid division, expected "train" or "eval"')

        normal = get_filedir(dir, True, base_dir)
        norm = [os.path.join(normal, file) for file in os.listdir(normal)]
        abnormal = get_filedir(dir, False, base_dir)
        abnorm = [os.path.join(abnormal, file) for file in os.listdir(abnormal)]


        return norm, abnorm

    def save_to_npz(self, destdir, div = 'train'):
        ''' Saves the data to a numpy file
            INPUT:
                destdir - string - the directory to save the data
                div - string - the division of the dataset to save
        '''
        normal, abnormal = self.load_data(div)
        destdir = os.path.join(destdir, div)
        os.makedirs(destdir, exist_ok=True)

        # Saving data in csv file too
        with open(os.path.join(destdir, 'data.csv'), 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['File', 'Label'])
            for file in normal:
                filename = file.split('/')[-1].split('.')[0] + '.npz'
                writer.writerow([filename, 0])
            for file in abnormal:
                filename = file.split('/')[-1].split('.')[0] + '.npz'
                writer.writerow([filename, 1])
                
        print("Converting Normal Files")
        for file in tqdm(normal):
            label = [0, 1]
            data = mne.io.read_raw_edf(file, preload=True, verbose='error')
            data = self.pipeline.apply(data)
            data = np.array(data.get_data())
            label = np.array(label)
            filename = file.split('/')[-1].split('.')[0] + '.npz'
            np.savez(os.path.join(destdir, filename), data=data, label=label)
        print("Converting Abnormal Files now")
        for file in tqdm(abnormal):
            label = [1, 0]
            data = mne.io.read_raw_edf(file, preload=True, verbose='error')
            data = self.pipeline.apply(data)
            data = np.array(data.get_data())
            label = np.array(label)
            filename = file.split('/')[-1].split('.')[0] + '.npz'
            np.savez(os.path.join(destdir, filename), data=data, label=label)

        return destdir





        
    

