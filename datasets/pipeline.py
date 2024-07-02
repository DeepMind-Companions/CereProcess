''' Defines all the preprocessing functions and applies them
    to the data in the pipeline
'''


import mne
import numpy as np
from .channels import CHANNELS, PAIRS

class Preprocess:
    def func(self, data):
        # Do something to the data
        # This is to be overloaded always
        return data
    def apply(self, data):
        ''' Applies the preprocessing pipeline to the data
            INPUT:
                data - EEG - data to be preprocessed
            OUTPUT:
                data - EEG - preprocessed data
        '''
        return self.func(data)

class ReduceChannels(Preprocess):
    ''' Reducing the number of channels to the 21 channels in use
        Takes in raw data in mne format
        Returns raw data in mne format with 21 channels only
    '''
    def __init__(self, channels=CHANNELS):
        self.channels = channels
    def func(self, data):
        return data.pick(self.channels)

class ClipData(Preprocess):
    ''' Responsible for Clipping the data inside a fixed voltage range
        Inputs: raw EEG data in MNE format
        Outputs: raw EEG data clipped between -absclipx10^-6 and absclipx10^-6
    '''
    def __init__(self, absclip):
        self.absclip = absclip
    def func(self, data):
        return data.apply_function(lambda data: np.clip(data, -0.000001*self.absclip, 0.000001*self.absclip))

class ResampleData(Preprocess):
    ''' Responsible for resampling the data to 100 Hz
        Inputs: raw EEG data in MNE format
        Outputs: raw EEG data resampled to 100 Hz
    '''
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
    def func(self, data):
        return data.resample(100)

class CropData(Preprocess):
    ''' Responsible for cropping the data to the specified time range
        Inputs: raw EEG data in MNE format
        Outputs: raw EEG data cropped to the specified time range
    '''
    def __init__(self, tmin, tmax):
        self.tmin = tmin
        self.tmax = tmax
    def func(self, data):
        return data.crop(tmin=self.tmin, tmax=self.tmax, include_tmax=False)

class HighPassFilter(Preprocess):
    ''' Responsible for applying a high pass filter to the data
        Inputs: raw EEG data in MNE format
        Outputs: raw EEG data with a high pass filter applied
    '''
    def __init__(self, l_freq, h_freq=None, fir_design='butterworth', l_trans_bandwidth='auto'):
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.fir_design = fir_design
        self.l_trans_bandwitdth=l_trans_bandwidth
    def func(self, data):
        return data.filter(l_freq=self.l_freq, h_freq=self.h_freq, fir_design=self.fir_design, l_trans_bandwidth=self.l_trans_bandwitdth)

class NotchFilter(Preprocess):
    ''' Responsible for applying a notch filter to the data
        Inputs: raw EEG data in MNE format
        Outputs: raw EEG data with a notch filter applied
    '''
    def __init__(self, freqs, fir_design='firwin'):
        self.freqs = freqs
        self.fir_design = fir_design
    def func(self, data):
        return data.notch_filter(self.freqs, fir_design=self.fir_design)

class BipolarRef(Preprocess):
    ''' Responsible for applying a bipolar reference to the data
        Inputs: raw EEG data in MNE format
        Outputs: raw EEG data with a bipolar reference applied
    '''
    def __init__(self, pairs=PAIRS):
        self.pairs = pairs
    def func(self, data):
        for anode, cathode in self.pairs:
            data = mne.set_bipolar_reference(data.load_data(), anode=[anode], cathode=[cathode], ch_name=f'{anode}-{cathode}', drop_refs=False, copy=True, verbose=False)
            data.drop_channels(ch_names=CHANNELS)   
            return data 



class Pipeline(Preprocess):
    ''' Pipeline class defines the preprocessing pipeline for the EEG data.
        Keeps the pipeline for preprocessing the data
    '''
    def __init__(self):
        ''' Constructor Function
            INPUT:
                pipeline - list - list of functions to be applied to the data
        '''
        
        self.pipeline = []

    def __iter__(self):
        ''' Returns the iterator for the pipeline
        '''
        return iter(self.pipeline)

    def add(self, func):
        ''' Adds a function to the pipeline
            INPUT:
                func - function - function to be added to the pipeline
        '''
        self.pipeline.append(func)

    def __add__(self, pipeline):
        ''' Adds a function to the pipeline
            INPUT:
                pipeline - function - function to be added to the pipeline
                pipeline - another list to be added to the pipeline
        '''
        new_pipeline = Pipeline()
        new_pipeline.pipeline = self.pipeline + pipeline.pipeline
        return new_pipeline

    def func(self, data):
        ''' Applies the pipeline to the data
            INPUT:
                data - EEG - data to be preprocessed
            OUTPUT:
                data - EEG - preprocessed data
        '''
        for func in self.pipeline:
            data = func.func(data)
        return data


