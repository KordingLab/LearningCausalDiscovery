import os
import numpy as np
from tqdm import tqdm
import pickle
from argparse import ArgumentParser
from collections import defaultdict
from sim2600 import params
from record_transistor_state import single_transistor_perturbation
from adjacency_matrix_generation import get_causal_effect

num_iterations = 128
step_limit = 30


def get_perturb_config(window, unique=False):
    # perturb at the middle of the interation_time
    perturb_timepoint = num_iterations // 2
    current_voltages = window[:, step_limit * perturb_timepoint]
    # 0: 'low', 1: 'high'
    perturb_types = np.where(current_voltages == 1, 0, 1)

    if unique:
        u, i = np.unique(window, axis=0, return_index=True)
        perturb_config = {tidx: (perturb_timepoint, perturb_types[tidx]) for tidx in i if window[tidx].std() != 0}
    else:
        perturb_config = {tidx: (perturb_timepoint, perturb_types[tidx]) for tidx in range(len(current_voltages))
                          if window[tidx].std() != 0}
    return perturb_config


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--game", type=str, choices=["Pitfall", "DonkeyKong", "SpaceInvaders"], default="DonkeyKong")
    parser.add_argument("--split", type=str, default="window_0_256")
    parser.add_argument("--file_path", type=str, default="DonkeyKong/HR/Regular_3510_step_81920.npy")
    parser.add_argument("--save_dir", type=str, default=".cache/sim_data")
    parser.add_argument("--unique", type=bool, default=False)
    args = parser.parse_args()
    game = args.game
    split = args.split
    root_dir = args.save_dir
    unique = args.unique

    game2rom = {"Pitfall": params.ROMS_PITFALL,
                "DonkeyKong": params.ROMS_DONKEY_KONG,
                "SpaceInvaders": params.ROMS_SPACE_INVADERS,
                }

    start, end = [int(number) for number in split.split("_")[1:]]
    set_range = {split: [start, end]}

    print(split, set_range[split])

    split_window = np.load(os.path.join(root_dir, "{}/HR/{}/Regular_3510_step_{}_rec_{}_window_{}_{}.npy".
                            format(game, split, num_iterations, step_limit, set_range[split][0],
                                set_range[split][1])), mmap_mode="r")

    splits = zip([split], [split_window])

    for name, split in splits:
        perturb_config = get_perturb_config(split, unique=unique)
        potential_effects = defaultdict(list)
        root_path = os.path.join(root_dir, "{}/HR/{}".format(game, name))

        for tidx, (halfclk, _action) in perturb_config.items():
            # do adaptive voltage single element lesion analysis
            path = os.path.join(root_path,
                                "Perturb_3510_step_{}_tidx_{}.npy".format(num_iterations + set_range[name][0], tidx))

            if os.path.exists(path):
                print("File {} exists, skip!".format(path))
                continue
            perturb = single_transistor_perturbation(tidx=tidx,
                                                     perturb_step=halfclk + set_range[name][0],
                                                     perturb_type=_action,
                                                     rom=game2rom[game],
                                                     num_iterations=num_iterations + set_range[name][0],
                                                     )

            np.save(path, perturb)
            print("Simulation end and save file at {}!".format(path))

        # calculate cause effect and save
        for tidx, (halfclk, _action) in tqdm(perturb_config.items(), total=len(perturb_config)):
            perturb = np.load(os.path.join(root_path, "Perturb_3510_step_{}_tidx_{}.npy".format(
                num_iterations + set_range[name][0], tidx)), mmap_mode="r")
            # padding to the fix length of one half-clock, last point is marker (-1)
            if perturb.shape[1] > step_limit:
                padded_perturb = perturb[:, :step_limit]
            else:
                padded_perturb = np.concatenate(
                    (perturb[:, :-1], np.tile(perturb[:, -2].reshape(-1, 1), step_limit - perturb.shape[1] + 1)),
                    axis=1)
            # compare the regular state and 'perturbed' state when the cause transistor (idx) is perturbed
            potential_effects[tidx] = get_causal_effect(perturb_config, padded_perturb, perturb_config[tidx][0], split,
                                                        tidx)
        # save the adjacency matrix
        with open(os.path.join(root_path, "adjacency_matrix.pkl"), "wb") as f:
            pickle.dump(potential_effects, f)
