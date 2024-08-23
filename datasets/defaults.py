from datasets.sc_pipeline import get_scratch_pl
import numpy as np

tuh_dataset = r'/home/system1/Arsalan/datasets_raw/TUH EEG Corpus'
nmt_dataset = r'/home/system1/Arsalan/datasets_raw/[M] nmt_scalp_eeg_dataset'


def get_def_ds(mins = 1):
    return (tuh_dataset, get_scratch_pl('TUH', mins), np.array([1, 1])), \
            (nmt_dataset, get_scratch_pl('NMT', mins), np.array([1900, 305]))


