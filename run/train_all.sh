# here configs has been set to different seeds
ROOT=run/configs
## model
CFG=($(find $ROOT -name "*.py"))
# device
GPUS=[0]
# parallel jobs
JOBS=2

# here exhibit how training is done on one card with one job iteratively
#for cfg in ${CFG[@]}; do
#  echo $cfg
#  python run/tools/cli/train.py --cfg=$CFG --gpu_ids=$GPUS
#done

# parallel training
CUDA_VISIBLE_DEVICES=1 parallel --link --linebuffer  --jobs=$JOBS python run/tools/cli/train.py --cfg={1} --gpu_ids=$GPUS ::: ${CFG[@]}