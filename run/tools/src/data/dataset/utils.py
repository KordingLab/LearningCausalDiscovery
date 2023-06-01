import torch


class TimeSeriesShiftAugmentation:
    """
    Implenmentation of the time series shift augmentation.
    """
    def __init__(self, shift_range=(-1200, 1200)):
        self.shift_range = shift_range

    def __call__(self, sequence):
        return self._shift_sequence(sequence)

    def _shift_sequence(self, sequence):
        steps = torch.randint(self.shift_range[0], self.shift_range[1] + 1, (1,)).item()

        if steps == 0:
            return sequence
        elif steps > 0:
            return torch.cat((sequence[steps:], sequence[-1].repeat(steps, 1)))
        else:
            return torch.cat((sequence[0].repeat(-steps, 1), sequence[:steps]))