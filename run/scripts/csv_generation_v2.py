# simplified csv file
import copy
import os
import pickle
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from joblib import Parallel, delayed

num_cores = 16
noise_seed = 42
seed = [42]

step_limit = 30
num_iterations = 128


set_range = {
    "train": (896, 1024),
    "val": (768, 896),
    "test": (640, 768),
    "extra": 'run/scripts/pretrain_windows.txt'
}

range2set = {v: k for k, v in set_range.items()}



def txt2tuple(txt_path):
    text = open(txt_path, "r").readlines()
    l = []
    for t in text:
        txt = t.strip().split('_')
        start, end = int(txt[-2]), int(txt[-1])
        l.append((start, end))
    return tuple(l)


def remove_constant_seqs(data, label, u_index=None):
    """
    create the dataframe of transistor pairs
    """
    if u_index is None:
        index = list(label.keys())
        data_dict = {i: data[i] for i in index}
    else:
        # only use the unique transistors to construct pairs
        data_dict = {i: data[i] for i in u_index}
        print(len(data_dict))
        # only use positive samples from windows

        # get the corresponding label a pair of transistors
        _label = {}
        for i in data_dict:
            for j in data_dict:
                if i == j:
                    continue
                if j in label[i]:
                    _label[(i, j)] = 1
                else:
                    _label[(i, j)] = 0


    # return the df have the transistor id and the label
    df = pd.DataFrame()
    df["id"] = range(len(_label))
    df["transistor_1"] = [key[0] for key, value in _label.items()]
    df["transistor_2"] = [key[1] for key, value in _label.items()]
    df["label"] = [value for key, value in _label.items()]
    return df


def generate_csv(root, extra=False, unique=True):
    _split = tuple([int(number) for number in root.split("/")[-1].split("_")[-2:]])
    _game = root.split("/")[-3]
    try:
        adjacency_matrix = pickle.load(open(os.path.join(root, "adjacency_matrix.pkl"), "rb"))
    except UnicodeDecodeError:
        adjacency_matrix = pickle.load(open(os.path.join(root, "adjacency_matrix.pkl"), "rb"), encoding="latin1")
    file_path = os.path.join(root, f"Regular_3510_step_{num_iterations}_rec_{step_limit}_window_{_split[0]}_{_split[1]}.npy")
    orig = np.load(file_path, mmap_mode="r")

    if extra:
        tag = 'extra'
    else:
        tag = range2set[_split]
    # remove all the repeated transistors
    if unique:
        u_index_path = os.path.join(root, f"Regular_3510_step_{num_iterations}_rec_{step_limit}_window_{_split[0]}_{_split[1]}_unique_index.pkl")
        if os.path.exists(u_index_path):
            u_index = pickle.load(open(u_index_path, "rb"))
        else:
            u, u_index = np.unique(orig, return_index=True, axis=0)
        if not os.path.exists(u_index_path):
            pickle.dump(u_index, open(u_index_path, "wb"))

        # if the number of unique transistors is less than 100, then skip this window
        if len(u_index) < 400:
            return pd.DataFrame()

        # 50/50 split for training/testing
        if tag == 'train' or 'tag' == 'extra':
            # inside first 1755 transistors
            u_index = [i for i in u_index if i < 1755]
        else:
            # inside last 1755 transistors
            u_index = [i for i in u_index if i >= 1755]

    # not consider the transistors with all 0 or all 1
    if unique:
        print("total unique transistors: {}".format(len(u_index)))
    else:
        print("total non-constant transistors: {}".format(len(adjacency_matrix)))

    df = remove_constant_seqs(
        orig, adjacency_matrix, u_index if unique else None
    )
    df['file_path'] = file_path
    orig_df = copy.deepcopy(df)
    print(df.label.value_counts())

    for _seed in seed:
        if tag == 'train':  # for training set
            # add the external data to the original data
            df_ext = get_external_data()
            df = pd.concat((df, df_ext), axis=0).reset_index(drop=True)
            # down sample the negative samples and oversample the positive samples
            df_pos = df[df.label == 1]
            df_neg = df[df.label == 0]
            print("number of positive samples: ", df_pos.shape, "number of negative samples: ", df_neg.shape)
            df_neg = df_neg.sample(n=int(len(df_pos)*train_up_sample_ratio), random_state=_seed)
            df = pd.concat((df_pos, df_neg), axis=0)

            print("after balancing:\n", df.label.value_counts())

        elif tag == 'extra':
            # filter all the non-i.i.d periods
            # if df.label.value_counts()[1] > 400:
            #     return pd.DataFrame()
            pass

        if not os.path.exists(os.path.join(root, f"csv/fold_seed_{_seed}")):
            os.makedirs(os.path.join(root, f"csv/fold_seed_{_seed}"))
        if tag == 'train':
            df.to_csv(os.path.join(root, f"csv/fold_seed_{_seed}/", f"{tag}_ds_{train_up_sample_ratio}_unique_{unique}.csv"), index=False)
            print(
                f"csv file saved to {os.path.join(root, f'csv/fold_seed_{_seed}/', f'{tag}_ds_{train_up_sample_ratio}_unique_{unique}.csv')}")
        elif tag == 'extra':
            pass
        else:
            df.to_csv(os.path.join(root, f"csv/fold_seed_{_seed}/", f"{tag}_ds_1.0_unique_{unique}.csv"), index=False)
            print(
                f"csv file saved to {os.path.join(root, f'csv/fold_seed_{_seed}/', f'{tag}_ds_1.0_unique_{unique}.csv')}")

        if extra:
            return df


def process_range(start_end_tuple):
    start, end,  = start_end_tuple
    df = generate_csv(os.path.join(root_dir, f"DonkeyKong/HR/window_{start}_{end}"), extra=True)
    return df


def get_external_data():
    # get the external data
    df_ext = []
    if not set_range['extra']:
        return pd.DataFrame()

    if isinstance(set_range['extra'][0], int):
        df_ext = generate_csv(os.path.join(root_dir, f"DonkeyKong/HR/window_{set_range['extra'][0]}_{set_range['extra'][1]}"), extra=True)
    elif isinstance(set_range['extra'][0], tuple):
        df_ext = Parallel(n_jobs=num_cores)(delayed(process_range)(start_end) for start_end in set_range['extra'])
        df_ext = pd.concat(df_ext, axis=0).reset_index(drop=True)
    return df_ext


def get_noised_data(root, tag, _seed=42):
    noise_scale = [0.03, 0.05, 0.1]
    start, end = set_range[tag]
    file_path = os.path.join(root, f"HR/window_{start}_{end}/"
                                  f"Regular_3510_step_{num_iterations}_rec_{step_limit}_window_{start}_{end}.npy")
    orig = np.load(file_path, mmap_mode="r")
    np.random.seed(_seed)

    for scale in noise_scale:
        noised_orig = copy.deepcopy(orig).astype(np.float64)
        # voltage value is limited in [0, 1], so not need to scale the noise for normalization
        noised_orig = noised_orig + scale * np.random.normal(0, 1.0, size=noised_orig.shape)

        np.save(os.path.join(root, f"HR/window_{start}_{end}/"
                                   f"Regular_3510_step_{num_iterations}_rec_{step_limit}_window_{start}_{end}_Noise_{scale}.npy"), noised_orig)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save_dir", type=str, default=".cache/sim_data/")
    parser.add_argument("--train_up_sample_ratio", type=float, default=3)

    args = parser.parse_args()
    root_dir = args.save_dir
    # misleading name, actually it is the ratio of negative samples to positive samples
    train_up_sample_ratio = args.train_up_sample_ratio

    if isinstance(set_range['extra'], str):
        set_range['extra'] = txt2tuple(set_range['extra'])

    # generate the train/valid/test csv for DonkeyKong
    # train:  896 - 40960 clk, val: 768 - 896 clk, test: 0-768 clk
    for split in ["train", "val", "test"]:
        files_dir = os.path.join(root_dir, f"DonkeyKong/HR/window_{set_range[split][0]}_{set_range[split][1]}")
        generate_csv(files_dir)

    # 5 snippets test data for official reported test
    for game in ["DonkeyKong", "Pitfall", "SpaceInvaders"]:
        for test_split in ((0, 128), (128, 256), (384, 512), (512, 640), (640, 768)):
            set_range['test'] = test_split
            range2set = {v: k for k, v in set_range.items()}

            files_dir = os.path.join(root_dir, f"{game}/HR/window_{test_split[0]}_{test_split[1]}")
            generate_csv(files_dir)
            # generate the additive noise data
            get_noised_data(os.path.join(root_dir, game), "test", _seed=noise_seed)




