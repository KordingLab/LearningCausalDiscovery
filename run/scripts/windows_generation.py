import os
import numpy as np
from tqdm import tqdm
from csv_generation import resample
from argparse import ArgumentParser

num_iterations = 256
overlapping = 128
step_limit = 400


def generate_windows(data):
    windows = {}
    window_size = num_iterations * step_limit
    overlapping_size = overlapping * step_limit
    for window_idx, i in enumerate(range(0, data.shape[1] - window_size, overlapping_size)):
        windows[window_idx] = data[:, i:i+window_size]
    return windows


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--game",
        type=str,
        choices=["Pitfall", "DonkeyKong", "SpaceInvaders", "All"],
        default="All",
    )
    parser.add_argument("--length", type=int, default=2048)
    parser.add_argument("--save_dir", type=str, default=".cache/sim_data/")

    args = parser.parse_args()
    game = args.game
    length = args.length
    root_dir = args.save_dir

    if game == "All":
        games = ["DonkeyKong", "Pitfall", "SpaceInvaders"]
    else:
        games = [game]

    for game in games:
        data = np.load(os.path.join(root_dir, f"{game}/HR/Regular_3510_step_{length}.npy"), mmap_mode="r")
        data = resample(data)
        windows = generate_windows(data)
        for window_name, window in tqdm(windows.items(), desc="save windows..."):
            np.save(os.path.join(root_dir, f"{game}/HR/Regular_3510_step_{num_iterations}_rec_{step_limit}_window_{window_name}.npy"), window)
