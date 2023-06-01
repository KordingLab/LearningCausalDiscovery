import os
import numpy as np
from argparse import ArgumentParser
from simulation_v4 import resample

num_iterations = 128
step_limit = 30



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--file_path", type=str, default="DonkeyKong/HR/Regular_3510_step_81920.npy")
    parser.add_argument("--save_dir", type=str, default=".cache/sim_data")

    args = parser.parse_args()
    game = args.file_path.split("/")[0]
    file_path = args.file_path
    root_dir = args.save_dir


    total_length = int(file_path.split("_")[-1].split(".")[0])
    splits = ["window_{}_{}".format(i, i+num_iterations) for i in range(0, total_length, num_iterations)]

    orig = np.load(os.path.join(args.save_dir, args.file_path), mmap_mode="r")
    # padding to sequence with fix length in a half-clock by the markers (-1) in recording

    marked_data = np.where(orig == 255, -1, orig)
    # detect the end of , last marker point is the end of the sequence
    marked_data = marked_data[:, :np.where(marked_data[0] != -1)[0][-1] + 2]
    # before padding
    print('raw sample length: ', marked_data.shape[1])
    clocks = []
    snippet_lengths = []
    markers = np.where(marked_data[0] == -1)[0].tolist()
    print(len(markers), 'half-clocks in total')

    for split in splits:
        start, end = [int(number) for number in split.split("_")[1:]]
        if start == 0:
            split_window = marked_data[:, :markers[end-1]+1]
        else:
            split_window = marked_data[:, markers[start - 1] + 1:markers[end-1]+1]
        set_range = {split: [start, end]}

        print(split, set_range[split])

        split_window = resample(split_window, n_steps=step_limit)
        print(split_window.shape)
        assert split_window.shape[1] == num_iterations*step_limit
        assert -1 not in split_window[0] or 255 not in split_window[0]

        # save the window data
        if not os.path.exists(os.path.join(root_dir, "{}/HR/{}".format(game, split))):
            os.makedirs(os.path.join(root_dir, "{}/HR/{}".format(game, split)))
        np.save(os.path.join(root_dir, "{}/HR/{}/Regular_3510_step_{}_rec_{}_window_{}_{}.npy".
                             format(game, split, num_iterations, step_limit, set_range[split][0],
                                    set_range[split][1])), split_window)
        print("saved: ", os.path.join(root_dir, "{}/HR/{}/Regular_3510_step_{}_rec_{}_window_{}_{}.npy".
                              format(game, split, num_iterations, step_limit, set_range[split][0],
                                     set_range[split][1])))