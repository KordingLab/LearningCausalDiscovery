from torch.utils.data import DataLoader, Dataset


def cycle(iterable):
    # cycle through an iterable
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


class CombinedCycleDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.max_len = max([len(dataset) for dataset in datasets])

    def __getitem__(self, index):
        dataset_idx = index % len(self.datasets)
        inner_idx = index // len(self.datasets)
        return self.datasets[dataset_idx][inner_idx % len(self.datasets[dataset_idx])]

    def __len__(self):
        return len(self.datasets) * self.max_len

