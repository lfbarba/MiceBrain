# Code path is the first argument
ROOT_PATH=$1
CODE_PATH=$1/MiceBrain
# python /myhome/MiceBrain/mice/training/conditional_training.py --batch_size=100 --epochs=200 --learning_rate=1e-4 --exp_name=ContidionalMiceBrain --load_checkpoint=checkpoints/ContidionalMiceBrain.pt --dataset_path=/mydata/chip/luisb/mice_brain/sections --im_size=128 --wandb
python $CODE_PATH/mice/training/conditional_training.py \
    --batch_size=100 \
    --epochs=200 \
    --learning_rate=1e-4 \
    --exp_name=ContidionalMiceBrainFull \
    --load_checkpoint=checkpoints/ContidionalMiceBrainFull.pt \
    --dataset_path=/s3/mlibra/mlibra-data/merfish/sections \
    --im_size=128 \
    --wandb
