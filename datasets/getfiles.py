import os
import random
from datetime import datetime

random.seed(datetime.now().timestamp())

def get_traineval(datapath, basedir):
    '''
        Using this to get the training and evaluation data from the source datapath seperately
        takes in the datapath for the MNE source files in the correct directory format and return train and eval path

        INPUT: 
            datapath - string - path to the MNE source files
            basedir - string - the directory before the train and eval directories

        OUPTUT: traindir, evaldir - string - path to the training and evaluation data
    '''
    traindir = os.path.join(datapath, basedir, 'train')
    evaldir = os.path.join(datapath, basedir, 'eval')

    return traindir, evaldir

def get_filedir(datapath, normal=True, basedir = '01_tcp_ar'):
    '''
        Using this to get the file directories for the MNE source files in the correct directory based on whether
        Normal or Abnormal Data
        takes in the datapath for the MNE source files in the correct directory format and returns the directory containing all files

        INPUT:
            datapath - string - path to the MNE source files (Train or Eval)
            normal - boolean - whether to get the normal or abnormal files
            basedir - string - the directory at the end of the normal or abnormal directory
            
        OUPTUT: filedir - directory containing the respective files (normal or abnormal)
    '''
    if normal:
        filedir = os.path.join(datapath, 'normal', basedir)
    else:
        filedir = os.path.join(datapath, 'abnormal', basedir)
   
    return filedir

def get_files(datapath, eval=False):
    '''
        Using this to get individual files along with their Y (output) values

        INPUT:
            filedir - string - path to the directory containing the files
        OUPTUT:
            files - list - list of files
            Y - list - list of output values

    '''

    # Get the training and evaluation directories
    traindir, evaldir = get_traineval(datapath)

    # Get the normal and abnormal directories
    normal_train = get_filedir(traindir)
    abnormal_train = get_filedir(traindir, False)
    normal_eval = get_filedir(evaldir)
    abnormal_eval = get_filedir(evaldir, False)

    # Get the files
    files = []

    # Get the normal files
    if (not eval):
        for file in os.listdir(normal_train):
            files.append((os.path.join(normal_train, file), 0))
        for file in os.listdir(abnormal_train):
            files.append((os.path.join(abnormal_train, file), 1))
    else:
        for file in os.listdir(normal_eval):
            files.append((os.path.join(normal_eval, file), 0))
        for file in os.listdir(abnormal_eval):
            files.append((os.path.join(abnormal_eval, file), 1))


    # Shuffle the files
    random.shuffle(files)

    return files



    
