GAME=(DonkeyKong Pitfall SpaceInvaders)

## generate all the simulation without perturbation
for i in "${GAME[@]}"
do
    python nmos_simulation/causal_simulation/simulation_v4.py --game=$i --num_iter=1024
done

WINDOWS=(window_256_512 window_512_768 window_768_1024)
TEST_WINDOW=window_768_1024
# generate all the perturbed data for DonkeyKong
for i in "${WINDOWS[@]}"
do
    python nmos_simulation/causal_simulation/split_generation.py --game=DonkeyKong \
      --file_path=.cache/sim_data/DonkeyKong/HR/Regular_3510_step_1024.npy --split=$i
done

# generate perturbed test data for other games
python nmos_simulation/causal_simulation/split_generation.py --game=Pitfall \
  --file_path=Pitfall/HR/Regular_3510_step_1024.npy --split=$TEST_WINDOW
python nmos_simulation/causal_simulation/split_generation.py --game=SpaceInvaders \
  --file_path=SpaceInvaders/HR/Regular_3510_step_1024.npy --split=$TEST_WINDOW
