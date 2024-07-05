"""dataset includes the classes and functions necessary to load and preprocess the dataset.

"""
import mne
import csv
import numpy as np
from tqdm.notebook import tqdm
import os
from .getfiles import get_files
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
        self.trainfiles, self.evalfiles = get_files(datapath, basedir)
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


    def save_to_npz(self, destdir, div = 'train', appendname=""):
        ''' Saves the data to a numpy file
            INPUT:
                destdir - string - the directory to save the data
                div - string - the division of the dataset to save
        '''
        destdir = os.path.join(destdir, div)
        os.makedirs(destdir, exist_ok=True)

        if (div == 'train'):
            normal = self.trainfiles['normal']
            abnormal = self.trainfiles['abnormal']
        elif (div == 'eval'):
            normal = self.evalfiles['normal']
            abnormal = self.evalfiles['abnormal']
        else:
            raise ValueError("Invalid division")

        # Saving data in csv file too

        if (appendname==""):
            with open(os.path.join(destdir, 'data.csv'), 'w') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['File', 'Label'])

        with open(os.path.join(destdir, 'data.csv'), 'a') as csvfile:
            writer = csv.writer(csvfile)

            print("Converting Normal Files")
            for file in tqdm(normal):
                label = [1, 0]
                try:
                    data = mne.io.read_raw_edf(file, preload=True, verbose='error')
                
                    data = self.pipeline.apply(data)
                    data = np.array(data.get_data())
                    label = np.array(label)
                    filename = file.split('/')[-1].split('.')[0] + appendname + '.npz'
                    np.savez(os.path.join(destdir, filename), data=data, label=label)
                    writer.writerow([filename, 0])
                except:
                    continue

            print("Converting Abnormal Files now")
            for file in tqdm(abnormal):
                label = [1, 0]
                try:
                    data = mne.io.read_raw_edf(file, preload=True, verbose='error')
                
                    data = self.pipeline.apply(data)
                    data = np.array(data.get_data())
                    label = np.array(label)
                    filename = file.split('/')[-1].split('.')[0] + appendname + '.npz'
                    np.savez(os.path.join(destdir, filename), data=data, label=label)
                    writer.writerow([filename, 1])
                except:
                    continue

        return destdir





        
    

