import torch
import numpy as np
import scipy.io as scio
from ..builder import DATASETS

from .utils import TimeSeriesShiftAugmentation

# import external modules
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))),
                             'external'))

from SRU_for_GCI.utils.utilFuncs import loadTrainingData, loadTrueNetwork


@DATASETS.register_module()
class Dream3(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root=None,
        max_length=1000,
        percentage=1.0,
        shift_range=None,
    ):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.subject = self.data_root[0].split("/")[-1].strip(".pt")
        self.check_files()
        if shift_range is not None:
            self.transform = TimeSeriesShiftAugmentation(shift_range)

    def check_files(self):
        # if isinstance(self.percentage, float):
        #     self.percentage = [0, self.percentage]

        if self.data_root is None or len(self.data_root) != 2:
            raise Exception("Invalid dataset path, should be a tuple of (seq_path, net_path)")

        # data_root = (seq_path, net_path)
        Xtrain = loadTrainingData(self.data_root[0], 'cpu')

        self.seqs = Xtrain
        n, l = Xtrain.shape
        Gref = loadTrueNetwork(self.data_root[1], n).T

        # Load data
        # duration = 21
        # self.examples = []
        # for i in range(n):
        #     for j in range(n):
        #         # skip the diag elements
        #         if i == j:
        #             continue
        #         # parse it into 46 snippets of sequences (the length of each one is 21 )
        #         for start in range(0, l, duration):
        #             self.examples.append(
        #                 {
        #                     "cause": i,
        #                     "effect": j,
        #                     "start_index": start,
        #                     "end_index": start + duration,
        #                     "label": 1 if Gref[i, j] != 0 else 0,
        #                 }
        #             )

        # percentage
        # self.examples = self.examples[int(len(self.examples) * self.percentage[0]):int(len(self.examples) * self.percentage[1])]

        self.examples = []
        for i in range(n):
            for j in range(n):
                # skip the diag elements
                if i == j:
                    continue
                self.examples.append(
                    {
                        "cause": i,
                        "effect": j,
                        "label": 1 if Gref[i, j] != 0 else 0,
                    }
                )


    def __getitem__(self, idx):
        example = self.examples[idx]
        seq = np.stack(
            [
                self.seqs[
                example["cause"]
                ],
                self.seqs[
                example["effect"]
                ],
            ],
            axis=-1,
        )
        seq = torch.tensor(seq, dtype=torch.float32)

        if hasattr(self, "transform"):
            seq = self.transform(seq)

        label = torch.tensor(example["label"], dtype=torch.int64)
        return seq, label

    def __len__(self):
            return len(self.examples)

