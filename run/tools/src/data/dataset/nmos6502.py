import torch
import os
import numpy as np
import pandas as pd
from ..builder import DATASETS

from .utils import TimeSeriesShiftAugmentation


@DATASETS.register_module()
class NMOS6502(torch.utils.data.Dataset):
    def __init__(
        self, data_root=None, split=None, interval=10, shift_range=None
    ):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.subject = data_root.split("/")[-1].strip(".npy")
        self.check_files()
        self.interval = interval
        if shift_range is not None:
            self.transform = TimeSeriesShiftAugmentation(shift_range)

    def check_files(self):
        if self.data_root is None:
            raise Exception("Invalid dataset path")

        # Load data
        self.time_series_df = pd.read_csv(self.split)

        # load all the files into memory
        windows = set(self.time_series_df["file_path"].values)
        self.windows = {window: np.load(window, mmap_mode="r") for window in windows}

    def __getitem__(self, idx):
        window = self.windows[self.time_series_df.iloc[idx]["file_path"]]
        cause_transistor = int(self.time_series_df.iloc[idx]["transistor_1"])
        effect_transistor = int(self.time_series_df.iloc[idx]["transistor_2"])

        seq = torch.tensor(np.stack(
                [
                    window[cause_transistor],
                    window[effect_transistor],
                ],
                axis=-1,
            )[:: self.interval, :], dtype=torch.float32)

        # augmentation
        if hasattr(self, "transform"):
            seq = self.transform(seq)

        label = torch.tensor(self.time_series_df.iloc[idx]["label"], dtype=torch.int64)

        return seq, label

    def __len__(self):
        return len(self.time_series_df)
