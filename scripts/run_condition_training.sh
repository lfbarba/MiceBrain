# Code path is the first argument
ROOT_PATH=$1
CODE_PATH=$1/MiceBrain
# python /myhome/MiceBrain/mice/training/conditional_training.py --batch_size=100 --epochs=200 --learning_rate=1e-4 --exp_name=ContidionalMiceBrain --load_checkpoint=checkpoints/ContidionalMiceBrain.pt --dataset_path=/mydata/chip/luisb/mice_brain/sections --im_size=128 --wandb
python $CODE_PATH/mice/training/conditional_training.py \
    --batch_size=100 \
    --epochs=200 \
    --learning_rate=1e-4 \
    --exp_name=maldi_128x1_channel_157 \
    --load_checkpoint=checkpoints/ContidionalMiceBrainFull.pt \
    --dataset_path=/s3/mlibra/mlibra-data/maldi/patches/merfish_128x1/ \
    --means-file=/s3/mlibra/mlibra-data/maldi/means_col_train.npy \
    --stds-file=/s3/mlibra/mlibra-data/maldi/stds_col_train.npy \
    --im_size=128 \
    --model_path=/mydata/mlibra/daniel/Diffusion_results/maldi/ \
    --wandb


python $CODE_PATH/mice/training/conditional_training.py \
    --batch_size=100\
    --epochs=200 \
    --learning_rate=1e-4 \
    --exp_name=merfish_128x1_channel_157 \
    --load_checkpoint=checkpoints/ContidionalMiceBrainFull.pt \
    --dataset_path=/s3/mlibra/mlibra-data/merfish/patches/merfish_128x1/ \
    --means-file=/s3/mlibra/mlibra-data/merfish/means_col_train.npy \
    --stds-file=/s3/mlibra/mlibra-data/merfish/stds_col_train.npy \
    --im_size=128 \
    --model_path=/mydata/mlibra/daniel/Diffusion_results/merfish/\
    --wandb
