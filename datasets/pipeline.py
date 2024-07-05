''' Defines all the preprocessing functions and applies them
    to the data in the pipeline
'''


import mne
import numpy as np
from .channels import CHANNELS, PAIRS, NMT_CHANNELS, NMT_PAIRS

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
        Outputs: raw EEG data clipped between 0 and absclipx10^-6
    '''
    def __init__(self, absclip):
        self.absclip = absclip
    def func(self, data):
        return data.apply_function(lambda data: np.clip(data, 0, 0.000001*self.absclip))


class ClipAbsData(Preprocess):
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
        sfreq = data.info['sfreq']
        if (sfreq == self.sample_rate):
            return data
        return data.resample(self.sample_rate)

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
    def __init__(self, l_freq, h_freq=None):
        self.l_freq = l_freq
        self.h_freq = h_freq
    def func(self, data):   
        iir_params = dict(order=4, ftype='butter', output='sos')
        return data.filter(l_freq=self.l_freq, h_freq=self.h_freq, method='iir', iir_params=iir_params, verbose='error')

class NotchFilter(Preprocess):
    ''' Responsible for applying a notch filter to the data
        Inputs: raw EEG data in MNE format
        Outputs: raw EEG data with a notch filter applied
    '''
    def __init__(self, freqs, fir_design='firwin'):
        self.freqs = freqs
        self.fir_design = fir_design
    def func(self, data):
        # Check the sampling rate and calculate the Nyquist frequency
        sfreq = data.info['sfreq']
        nyquist_freq = sfreq / 2
        if (self.freqs < nyquist_freq):
            return data.notch_filter(self.freqs, fir_design=self.fir_design, verbose='error')
        return data

class BipolarRef(Preprocess):
    ''' Responsible for applying a bipolar reference to the data
        Inputs: raw EEG data in MNE format
        Outputs: raw EEG data with a bipolar reference applied
    '''
    def __init__(self, pairs=PAIRS, channels=CHANNELS):
        self.pairs = pairs
        self.channels=channels
    def func(self, data):
        for anode, cathode in self.pairs:
            data = mne.set_bipolar_reference(data.load_data(), anode=[anode], cathode=[cathode], ch_name=f'{anode}-{cathode}', drop_refs=False, copy=True, verbose=False)
        data.drop_channels(ch_names=self.channels)   
        return data 

class Reverse(Preprocess):
    ''' Responsible for reversing the data
        Inputs: raw EEG data in MNE format
        Outputs: raw EEG data reversed
    '''
    def func(self, data):
        # Get the data as a NumPy array
        data_only, _ = data[:]

        # Reverse the time sequence of the data
        data_reversed = np.flip(data_only, axis=1)

        # Create a new Raw object with the reversed data
        info = data.info
        raw_reversed = mne.io.RawArray(data_reversed, info, verbose='error')
        return raw_reversed

class MakeNormal(Preprocess):
    ''' Responsible for making the data normal
        Inputs: raw EEG data in MNE format
        Outputs: raw EEG data made normal
    '''
    def func(self, data):
        return data.apply_function(lambda data: (data - np.mean(data)) / np.std(data))

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


def get_wavenet_pipeline(dataset = 'TUH'):
    ''' Returns the preprocessing pipeline for the Wavenet model
    '''
    pipeline = Pipeline()
    if (dataset == 'TUH'):
        pipeline.add(ReduceChannels())
        pipeline.add(BipolarRef())
    elif (dataset == 'NMT'):
        pipeline.add(ReduceChannels(channels= NMT_CHANNELS))
        pipeline.add(BipolarRef(pairs=NMT_PAIRS, channels= NMT_CHANNELS))
    pipeline.add(ClipData(100))
    pipeline.add(ResampleData(250))
    pipeline.add(CropData(0, 60))
    pipeline.add(HighPassFilter(1.0))
    pipeline.add(NotchFilter(60))
    return pipeline

def get_wavenet_reverse(dataset='TUH'):
    ''' Returns the preprocessing pipeline for the Wavenet model (second min)
    '''
    pipeline = Pipeline()
    if (dataset == 'TUH'):
        pipeline.add(ReduceChannels())
        pipeline.add(BipolarRef())
    elif (dataset == 'NMT'):
        pipeline.add(ReduceChannels(channels= NMT_CHANNELS))
        pipeline.add(BipolarRef(pairs=NMT_PAIRS, channels= NMT_CHANNELS))
    pipeline.add(ClipData(100))
    pipeline.add(ResampleData(250))
    pipeline.add(CropData(60, 120))
    pipeline.add(HighPassFilter(1.0))
    pipeline.add(NotchFilter(60))
    pipeline.add(Reverse())
    return pipeline


def get_scnet_pipeline():
    '''Returns the preprocessing pipeline for SCNet Model
    '''
    pipeline = Pipeline()
    pipeline.add(ReduceChannels())
    pipeline.add(ResampleData(100))
    pipeline.add(CropData(60, 480))
    pipeline.add(ClipData(100))
    pipeline.add(BipolarRef())
    return pipeline

