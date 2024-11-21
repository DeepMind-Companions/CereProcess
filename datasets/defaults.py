from datasets.sc_pipeline import get_scratch_pl
from datasets.sc_pipeline import get_scratch_updated_pl

import numpy as np

tuh_dataset = r'/home/ali/arsalan/Data/TUH EEG Corpus'
nmt_dataset = r'/home/ali/arsalan/Data/[M] nmt_scalp_eeg_dataset'
nmt_new_dataset = r'/media/dll-1/SSD 4TB/EEG Datasets/nmt_v2_scalp_eeg_dataset/'

def get_def_ds(mins = 1):
    return (tuh_dataset, get_scratch_pl('TUH', mins), np.array([1, 1]), "results/tuh"), \
            (nmt_dataset, get_scratch_updated_pl('NMT', mins), np.array([1900, 305]), "results/nmt"), \
            (nmt_new_dataset, get_scratch_updated_pl('NMT', mins), np.array([3945, 708]), "results/nmtv2")

