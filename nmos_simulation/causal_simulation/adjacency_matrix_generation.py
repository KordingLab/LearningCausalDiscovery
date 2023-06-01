import os
import numpy as np
from tqdm import tqdm
import pickle
from collections import defaultdict
from argparse import ArgumentParser
from simulation_v3 import resample
from glob import glob

# total simulation steps
num_iterations = 512
# interpolation for each clock
step_limit = 30
# perturb times (only work for multiple perturb)
perturb_times = 8


def get_config(multiple, root_dir='.cache/sim_data'):
    # get the config recorded how and when to perturb which transistor
    if multiple:
        with open(os.path.join(root_dir, "{}/multiple_{}_perturb_config.pkl".format(game, perturb_times)), "rb") as f:
            config = pickle.load(f)
    else:
        with open(os.path.join(root_dir, "{}/perturb_config.pkl".format(game)), "rb") as f:
            config = pickle.load(f)
    return config


def get_orig(game, root_dir):
    if os.path.exists(os.path.join(root_dir, "{}/HR/Regular_3510_step_{}_rec_{}.npy".format(game, num_iterations, step_limit))):
        orig = np.load(os.path.join(root_dir, "{}/HR/Regular_3510_step_{}_rec_{}.npy".format(game, num_iterations, step_limit)), mmap_mode='r')
    else:
        orig = np.load(os.path.join(root_dir, "{}/HR/Regular_3510_step_{}_rec_{}.npy".format(game, num_iterations, step_limit)), mmap_mode='r')
        orig = resample(orig)
        np.save(os.path.join(root_dir, "{}/HR/Regular_3510_step_{}_rec_{}.npy".format(game, num_iterations, step_limit)), orig)
    return orig


def get_common_effects(l):
    # Remove all the empty sublists
    l = [sublist for sublist in l if sublist]
    # Convert the first sublist to a set
    if l:
        common_elements = set(l[0])

        # Iterate through the remaining sublists and compute the intersection
        for sublist in l[1:]:
            common_elements = common_elements.intersection(sublist)

        # Convert the result back to a list
        common_elements = list(common_elements)
    else:
        common_elements = []
    return common_elements


def get_causal_effect(unique_perturb, padded_perturb, perturb_timepoint, orig, idx):
    # compare the regular state and 'perturbed' state when the cause transistor (idx) is perturbed
    effects = []
    for i in unique_perturb:
        # skip the cause transistor itself and the transistor has constant value
        if orig[i].std() != 0 and i != idx:
            # detect the first point where the perturbation actually works
            causal_effect = np.abs((padded_perturb[i] - orig[i, perturb_timepoint*step_limit:(perturb_timepoint+1)*step_limit])).mean()
            if causal_effect > 0:
                effects.append(i)
    return effects


def get_all_effect_across_time(idx, orig, unique_perturb):
    # use glob to find perturb files of different perturb times
    files = glob(os.path.join(root_dir, "{}/HR/multiple_{}/Perturb_3510_step_{}_tidx_{}_halfclk_*.npy".
                              format(game, perturb_times, num_iterations, idx)))
    effects = []
    for perturb in files:
        perturbation_step = int(perturb.split('_')[-1].split('.')[0])
        perturb = np.load(perturb, mmap_mode='r')
        # padding to the fix length of one half-clock, last point is marker (-1)
        if perturb.shape[1] > step_limit:
            padded_perturb = perturb[:, :step_limit]
        else:
            padded_perturb = np.concatenate(
                (perturb[:, :-1], np.tile(perturb[:, -2].reshape(-1, 1), step_limit - perturb.shape[1] + 1)), axis=1)

        effects.append(get_causal_effect(unique_perturb, padded_perturb, perturbation_step, orig, idx))

    return effects


def cross_valid_on_labels(all_transistor_effects):
    total_intersections = 0
    total_train = 0
    for idx, effects in all_transistor_effects.items():
        train_effects = effects[:int(len(effects) * 0.8)]
        test_effects = effects[int(len(effects) * 0.8):]

        train_common_effects = get_common_effects(train_effects)
        test_common_effects = get_common_effects(test_effects)

        intersection = set(train_common_effects).intersection(set(test_common_effects))
        total_intersections += len(intersection)
        total_train += len(train_common_effects)
    consistency = total_intersections / total_train
    print('the consistency of label on the first 0.8 length and the last 0.2 length is {}'.format(consistency))


def get_label(game, root_dir, multiple=False, num_iterations=256):
    unique_perturb = get_config(multiple=multiple)
    potential_all_effects = defaultdict(list)
    potential_effects = defaultdict(list)
    orig = get_orig(game, root_dir)
    for idx in tqdm(unique_perturb):
        # load the perturbed data for transistor idx
        perturb = np.load(os.path.join(root_dir, "{}/HR/Perturb_3510_step_{}_tidx_{}.npy".format(game, num_iterations, idx)), mmap_mode='r')
        # padding to the fix length of one half-clock, last point is marker (-1)
        if perturb.shape[1] > step_limit:
            padded_perturb = perturb[:, :step_limit]
        else:
            padded_perturb = np.concatenate(
                (perturb[:, :-1], np.tile(perturb[:, -2].reshape(-1, 1), step_limit - perturb.shape[1] + 1)), axis=1)
        # compare the regular state and 'perturbed' state when the cause transistor (idx) is perturbed
        if not multiple:
            potential_effects[idx] = get_causal_effect(unique_perturb, padded_perturb, unique_perturb[idx][0], orig, idx)
        else:
            potential_all_effects[idx] = get_all_effect_across_time(idx, orig, unique_perturb)

            potential_effects[idx] = get_common_effects(potential_all_effects[idx])

    if multiple:
        with open(os.path.join(root_dir, "{}/all_adjacency_matrices.pkl".format(game)), "wb") as f:
            pickle.dump(potential_all_effects, f)
        # test consistency of the causality consistency
        cross_valid_on_labels(potential_all_effects)

    return potential_effects


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--game", type=str, choices=["Pitfall", "DonkeyKong", "SpaceInvaders", "All"], default="All")
    parser.add_argument("--multiple", type=bool, default=False)
    parser.add_argument("--save_dir", type=str, default=".cache/sim_data")

    args = parser.parse_args()
    game = args.game
    multiple = args.multiple
    root_dir = args.save_dir

    if game == "All":
        games = ["DonkeyKong", "Pitfall", "SpaceInvaders"]
    else:
        games = [game]

    for game in games:
        label = get_label(game, root_dir, multiple=multiple)
        with open(os.path.join(root_dir, "{}/adjacency_matrix.pkl".format(game)), "wb") as f:
            pickle.dump(label, f)

