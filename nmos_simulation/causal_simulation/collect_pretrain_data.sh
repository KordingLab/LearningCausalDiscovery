JOBS=8
## generate all the simulation without perturbation
python nmos_simulation/causal_simulation/simulation_v4.py --game=DonkeyKong --num_iter=40960
python nmos_simulation/causal_simulation/simulation_v4.py --game=Pitfall --num_iter=1024
python nmos_simulation/causal_simulation/simulation_v4.py --game=SpaceInvaders --num_iter=1024

# generate all the windows for these games
python nmos_simulation/causal_simulation/window_generation.py --file_path=DonkeyKong/HR/Regular_3510_step_40960.npy
python nmos_simulation/causal_simulation/windows_generation.py --file_path=Pitfall/HR/Regular_3510_step_1024.npy
python nmos_simulation/causal_simulation/windows_generation.py --file_path=SpaceInvaders/HR/Regular_3510_step_1024.npy

## read lines from a txt and store them in the WINDOWS
GAME=DonkeyKong
WINDOWS=()
while read line; do
    WINDOWS+=($line)
done < nmos_simulation/causal_simulation/perturb_windows.txt

# print the content of WINDOWS

## generate all the perturbed data for DonkeyKong
parallel --linebuffer --jobs=$JOBS \
  python nmos_simulation/causal_simulation/split_generation.py --game=$GAME \
  --file_path=DonkeyKong/HR/Regular_3510_step_81920.npy --split={1} --unique=True ::: ${WINDOWS[@]}

## generate the test data for Pitfall and SpaceInvaders
TEST_WINDOWS=(window_0_128 window_128_256 window_384_512 window_512_640 window_640_768 window_768_896 window_896_1024)
TEST_WINDOWS=(window_768_896 window_896_1024)
OTHER_GAMES=(Pitfall SpaceInvaders)

for GAME in ${OTHER_GAMES[@]};
do
  parallel --linebuffer --jobs=$JOBS \
    python nmos_simulation/causal_simulation/split_generation.py --game=$GAME \
    --file_path=$GAME/HR/Regular_3510_step_1024.npy --split={1} --unique=True ::: ${TEST_WINDOWS[@]}
done