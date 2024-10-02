from datasets.sc_pipeline import get_scratch_pl
from datasets.sc_pipeline import get_scratch_updated_pl

import numpy as np

tuh_dataset = r'/home/dll-1/Desktop/eeg/datasets/nmt_scalp_eeg_dataset/'
nmt_dataset = r'/home/dll-1/Desktop/eeg/datasets/nmt_scalp_eeg_dataset/'
nmt_new_dataset = r'/media/dll-1/SSD 4TB/EEG Datasets/nmt_v2_scalp_eeg_dataset/'

def get_def_ds(mins = 1):
    return (tuh_dataset, get_scratch_pl('TUH', mins), np.array([1, 1]), "/media/dll-1/SSD 4TB/EEG Datasets/athar arsalan deepmind/tuh_processed_results_deepmind/"), \
            (nmt_dataset, get_scratch_updated_pl('NMT', mins), np.array([1900, 305]), "/media/dll-1/SSD 4TB/EEG Datasets/athar arsalan deepmind/nmt_processed_results_deepmind/"), \
            (nmt_new_dataset, get_scratch_updated_pl('NMT', mins), np.array([3945, 708]), "/media/dll-1/SSD 4TB/EEG Datasets/athar arsalan deepmind/nmt_v2_processed_results_deepmind/")

