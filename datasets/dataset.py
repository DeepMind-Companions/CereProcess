"""dataset includes the classes and functions necessary to load and preprocess the dataset.

"""
import mne
import csv
import numpy as np
from tqdm.notebook import tqdm
import os
from .getfiles import get_files
from .pipeline import Pipeline, MultiPipeline

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

        # Calculating the lengths of both train and eval
        trainlen = [len(self.trainfiles['normal']), len(self.trainfiles['abnormal'])]
        evallen = [len(self.evalfiles['normal']), len(self.evalfiles['abnormal'])]
        # Also getting the fullpath
        fullpath = os.path.join(datapath, basedir)
        if len(fullpath) > 10:
            fullpath = fullpath[-10:]

        self.id = fullpath + '_T' + str(trainlen) + '_E' + str(evallen)

        self.pipeline = MultiPipeline()

    def get_id(self):
        ''' Returns the ID of the dataset
        '''
        return self.id

    def set_pipeline(self, pipeline):
        ''' Adds a pipeline to the dataset
            INPUT:
                pipeline - Pipeline - the pipeline to add
        '''
        if pipeline.__class__.__name__ == 'Pipeline':
            self.pipeline = MultiPipeline([pipeline])
        elif pipeline.__class__.__name__ == 'MultiPipeline':
            self.pipeline = pipeline
        else:
            raise ValueError("Invalid Pipeline")
            

    def add_pipeline(self, pipeline):
        ''' Adds a pipeline to the dataset
            INPUT:
                pipeline - Pipeline - the pipeline to add
        '''
        self.pipeline = self.pipeline + pipeline


    def save_to_npz(self, destdir, div = 'train'):
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

        totalpl = len(self.pipeline)
        print("Total Pipelines: ", totalpl)
        print("Total Normal Files: ", len(normal))
        print("Total Abnormal Files: ", len(abnormal))
        # Saving data in csv file too
        with open(os.path.join(destdir, 'data.csv'), 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['File', 'Label'])

            print("Converting Normal Files")
            for file in tqdm(normal):
                label = [1, 0]
                try:
                    data = mne.io.read_raw_edf(file, preload=True, verbose='error')
                
                    for i, pipeline in enumerate(self.pipeline):
                        try:
                            appendname = '_P' + str(i)
                            data = pipeline.apply(data)
                            # data = self.pipeline.apply(data)
                            data = np.array(data.get_data())
                            label = np.array(label)
                            filename = file.split('/')[-1].split('.')[0] + appendname + '.npz'
                            np.savez(os.path.join(destdir, filename), data=data, label=label)
                            writer.writerow([filename, 0])
                        except:
                            continue
                except:
                    continue

            print("Converting Abnormal Files now")
            for file in tqdm(abnormal):
                label = [1, 0]
                try:
                    data = mne.io.read_raw_edf(file, preload=True, verbose='error')
                
                    for i, pipeline in enumerate(self.pipeline):
                        try:
                            appendname = '_P' + str(i)
                            data = pipeline.apply(data)
                            data = np.array(data.get_data())
                            label = np.array(label)
                            filename = file.split('/')[-1].split('.')[0] + appendname + '.npz'
                            np.savez(os.path.join(destdir, filename), data=data, label=label)
                            writer.writerow([filename, 1])
                        except:
                            continue
                except:
                    continue

        return destdir





        
    

