import os
import pickle
import numpy as np
from tqdm import tqdm
from sim2600 import params
from argparse import ArgumentParser
from record_transistor_state import single_transistor_perturbation, record_regular_transistor_state

step_limit = 30


def resample(marked_data, n_steps=step_limit):
    # padding to sequence with fix length in a half-clock by the markers (-1) in recording
    marked_data = np.where(marked_data==255, -1, marked_data)
    # detect the end of , last marker point is the end of the sequence
    marked_data = marked_data[:, :np.where(marked_data[0]!=-1)[0][-1]+2]
    # before padding
    print('raw sample length: ', marked_data.shape[1])
    clocks = []
    snippet_lengths = []
    markers = np.where(marked_data[0]==-1)[0].tolist()
    for idx, marker in enumerate(markers):
        if idx == 0:
            clock = marked_data[:, :markers[idx]]
        else:
            clock = marked_data[:, markers[idx-1]+1:markers[idx]]
        steps = clock.shape[1]
        snippet_lengths.append(steps)
        # if the step number in a half-clock is less than step_limit, padding it with the last state to step_limit
        if steps < n_steps:
            if clock.shape[1] == 0:
                clock = np.concatenate((clock, np.tile(clock.reshape(-1, 1), n_steps-steps)), axis=1)
            else:
                clock = np.concatenate((clock, np.tile(clock[:, -1].reshape(-1, 1), n_steps-steps)), axis=1)
        clocks.append(clock)
    marked_data = np.concatenate(clocks, axis=1) if len(clocks) > 1 else clocks[0]
    print('maximum snippet length: ', max(snippet_lengths))
    return marked_data


def get_where_to_perturb(game, root_dir, iteration_time=256, n_perturb=1):
    print('Calculating where to record perturbations')
    if os.path.exists(os.path.join(root_dir, "{}/HR/Regular_3510_step_{}.npy".format(game, iteration_time))):
        # padding the sequence with fix length in a half-clock by the markers (-1) in recording
        if not os.path.exists(os.path.join(root_dir, "{}/HR/Regular_3510_step_{}_rec_{}.npy".format(game, iteration_time, step_limit))):
            orig = resample(np.load(os.path.join(root_dir, "{}/HR/Regular_3510_step_{}.npy".format(game, iteration_time)), mmap_mode='r'))
            np.save(os.path.join(root_dir, "{}/HR/Regular_3510_step_{}_rec_{}.npy".format(game, iteration_time, step_limit)), orig)
        else:
            orig = np.load(os.path.join(root_dir, "{}/HR/Regular_3510_step_{}_rec_{}.npy".format(game, iteration_time, step_limit)), mmap_mode='r')

        # multiple times of perturbation / only perturb the first time
        if n_perturb == 1:
            # perturb at the middle of the interation_time
            perturb_timepoint = iteration_time // 2
            current_voltages = orig[:, perturb_timepoint*step_limit]
            # 0: 'low', 1: 'high'
            perturb_types = np.where(current_voltages == 1, 0, 1)
            unique_perturb = {tidx: (perturb_timepoint, perturb_types[tidx]) for tidx in range(len(current_voltages))
                              if orig[tidx].std()!=0}

            with open(os.path.join(root_dir, "{}/perturb_config.pkl".format(game)), "wb") as f:
                pickle.dump(unique_perturb, f, protocol=2)

            return unique_perturb
        else:
            # find perturbation timepoints with fixed interval
            perturb_timepoints = [np.linspace(8, iteration_time, num=n_perturb, endpoint=False).tolist() for _ in orig]
            unique_perturbs = {}  # {idx: [(timepoint, type), (timepoint, type), ...]}
            for idx, seq in enumerate(orig):
                # only save the unique transistors and the non-constant transistors
                unique_perturbs[idx] = []
                if seq.std() == 0:
                    continue
                for t in perturb_timepoints[idx]:
                    # check the the last unit before perturb_timepoints in orig is high or low
                    current_voltage = seq[int(t)*step_limit]
                    if current_voltage == 1:
                        unique_perturbs[idx].append((int(t), 0))
                    else:
                        unique_perturbs[idx].append((int(t), 1))

            with open(os.path.join(root_dir, "{}/multiple_{}_perturb_config.pkl".format(game, n_perturb)), "wb") as f:
                pickle.dump(unique_perturbs, f, protocol=2)

            return unique_perturbs

    else:
        raise NotImplementedError, 'Calculating where to record perturbations needs the regular state to be recorded'



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--game", type=str, choices=["Pitfall", "DonkeyKong", "SpaceInvaders", "All"], default="All")
    parser.add_argument("--num_iter", type=int, default=256)
    parser.add_argument("--n_perturb", type=int, default=1)
    parser.add_argument("--action", type=str, choices=["Regular", "Perturb"], default="Regular")
    parser.add_argument("--save_dir", type=str, default=".cache/sim_data")

    args = parser.parse_args()
    game = args.game
    n_perturb = args.n_perturb
    iteration_time = args.num_iter
    action = args.action
    print(args)
    root_dir = args.save_dir
    game2rom = {"Pitfall": params.ROMS_PITFALL,
                "DonkeyKong": params.ROMS_DONKEY_KONG,
                "SpaceInvaders": params.ROMS_SPACE_INVADERS,
                }

    if game == "All":
        games = ["DonkeyKong", "Pitfall", "SpaceInvaders"]
    else:
        games = [game]

    for game in games:
        if not os.path.exists(os.path.join(root_dir, "{}/HR/".format(game))):
            os.makedirs(os.path.join(root_dir, "{}/HR/".format(game)))
        path = os.path.join(root_dir, "{}/HR/{}_3510_step_{}.npy".format(game, action, iteration_time))
        if not os.path.exists('/'.join(path.split('/')[:-1])):
            os.makedirs('/'.join(path.split('/')[:-1]))

        print("Simulation start!")
        if action != "Regular":
            # get unique transistor index to reduce the number of simulation
            # get where actually to start recording the perturbed data
            if n_perturb == 1:
                perturb_config_path = os.path.join(root_dir, "{}/perturb_config.pkl".format(game))
            else:
                perturb_config_path = os.path.join(root_dir, "{}/multiple_{}_perturb_config.pkl".format(game, n_perturb))
            if os.path.exists(perturb_config_path):
                with open(os.path.join(perturb_config_path), "rb") as f:
                    perturb_config = pickle.load(f)
            else:
                perturb_config = get_where_to_perturb(game, root_dir, iteration_time=iteration_time, n_perturb=n_perturb)

            if n_perturb == 1:
                for tidx, (halfclk, _action) in perturb_config.items():
                    # do adaptive voltage single element lesion analysis
                    data = single_transistor_perturbation(tidx=tidx,
                                                          perturb_step=halfclk,
                                                          perturb_type=_action,
                                                          rom=game2rom[game],
                                                          num_iterations=iteration_time)

                    try:
                        path = os.path.join(root_dir, "{}/HR/{}_3510_step_{}_tidx_{}.npy".format(game, action, iteration_time, tidx))
                        np.save(path, data)
                        print("Simulation end and save file at {}!".format(path))
                    except LookupError as e:
                        print("Simulation end but save file failed!\n{e}")
            else:
                for tidx, perturb_points in perturb_config.items():
                    print(perturb_points)
                    for halfclk, _action in perturb_points:
                        data = single_transistor_perturbation(tidx=tidx,
                                                              perturb_step=halfclk,
                                                              perturb_type=_action,
                                                              rom=game2rom[game],
                                                              num_iterations=iteration_time)
                        try:
                            if not os.path.exists(os.path.join(root_dir, "{}/HR/multiple_{}/".format(game, n_perturb))):
                                os.makedirs(os.path.join(root_dir, "{}/HR/multiple_{}/".format(game, n_perturb)))
                            path = os.path.join(root_dir, "{}/HR/multiple_{}/{}_3510_step_{}_tidx_{}_halfclk_{}.npy".format(game, n_perturb, action, iteration_time, tidx, halfclk))
                            np.save(path, data)
                            print("Simulation end and save file at {}!".format(path))
                        except LookupError as e:
                            print("Simulation end but save file failed!\n{e}")
        else:
            # Collect regular simulation data
            data = record_regular_transistor_state(rom=game2rom[game],
                                                   num_iterations=iteration_time)
            try:
                np.save(path, data)
                print("Simulation end and save file at {}!".format(path))
            except LookupError as e:
                print("Simulation end but save file failed!\n{e}")


