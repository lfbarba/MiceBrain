import sys
import os

from argparse import ArgumentParser

import torch
from pytorch_base.experiment import PyTorchExperiment
from pytorch_base.base_loss import BaseLoss

import random
from tqdm.auto import tqdm

from diffusers import UNet1DModel
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
import math
from mice.datasets.slice_dataset import SliceDataset
import numpy as np

from torch.utils.data import Dataset

def get_dataset(kwargs):
    dataset = SliceDataset(**kwargs)
    torch.manual_seed(0)
    perm = torch.randperm(len(dataset))
    trainSet = torch.utils.data.Subset(dataset, perm[:round(0.95 * len(dataset))])
    testSet = torch.utils.data.Subset(dataset, perm[round(0.95 * len(dataset)):])
    return trainSet, testSet

class diffusion_loss(BaseLoss):

    def __init__(self):
        stats_names = ["loss"]
        super(diffusion_loss, self).__init__(stats_names)



    def compute_loss(self, instance, model):
        mse = torch.nn.MSELoss()
        x = instance.flatten(0, 1)
        target_channel = 145
        source_channel = -1

        x_0 = x[:, target_channel].to(device)
        conditioning = x[:, source_channel].to(device)
        # We train with guidance-free
        conditioning[:len(conditioning)//2] *= 0

        noise = torch.randn_like(x_0).to(device)
        bs = x_0.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bs,), device=x_0.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        x_t = noise_scheduler.add_noise(x_0, noise, timesteps)
        input = torch.stack([conditioning, x_t], dim=1)


        model.zero_grad()
        noise_pred = model(input, timesteps, return_dict=False)[0]

        loss = mse(noise_pred, noise)
        return loss, {"loss": loss}


def load_model(model, model_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"models loaded from checkpoint {model_path}")


if __name__ == '__main__':
    import lovely_tensors as lt

    lt.monkey_patch()

    parser = ArgumentParser(description="PyTorch experiments")
    parser.add_argument("--batch_size", default=50, type=int, help="batch size of every process")
    parser.add_argument("--epochs", default=50, type=int, help="number of epochs to train")
    parser.add_argument("--learning_rate", default=0.0001, type=float, help="learning rate")
    parser.add_argument("--weight_decay", default=0.001, type=float, help="weight decay")
    parser.add_argument("--scheduler", default="[500]", type=str, help="scheduler decrease after epochs given")
    parser.add_argument("--lr_decay", default=0.1, type=float, help="Learning rate decay")
    parser.add_argument("--exp_name", default='random_experiment', type=str, help="Experiment name")
    parser.add_argument('--wandb', action='store_true', help="Use wandb")
    parser.add_argument("--load_checkpoint", default='', type=str, help="name of models in folder checkpoints to load")
    parser.add_argument("--seed", default=-1, type=int, help="Random seed")

    parser.add_argument("--dataset_path", type=str, help="Path to the dataset file or folder")
    parser.add_argument("--im_size", type=int, default=512, help="In the case of tiff, the size of the crops to split the tiff files")
    parser.add_argument("--rescale", type=int, default=512, help="The side length of the images in the dataset")


    args = vars(parser.parse_args())
    temp = args["scheduler"].replace(" ", "").replace("[", "").replace("]", "").split(",")
    args["scheduler"] = [int(x) for x in temp]
    args["seed"] = random.randint(0, 20000) if args["seed"] == -1 else args["seed"]

    means = torch.tensor(np.load("means.npy"))
    stds = torch.tensor(np.load("stds.npy"))
    kwargs = {
        "path": args['dataset_path'],
        'im_size':args['im_size'],
        "train_transform": True,
        'stds': stds,
        'means': means
    }
    trainSet, testSet = get_dataset(kwargs)


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps')
    # device = torch.device('cpu')
    print(device)
    model_path = f"checkpoints/{args['exp_name']}.pt"

    model = UNet1DModel(
        sample_size=args['im_size'],  # Adjusted to im_size
        in_channels=2,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(32, 64),  # Reduced the number of output channels per block
        down_block_types=(
            "DownBlock1D",  # Keep the same
            "AttnDownBlock1D",  # One less downsampling block
        ),
        up_block_types=(
            "AttnUpBlock1D",  # Keep the same
            "UpBlock1D",  # Reduced the number of upsampling blocks accordingly
        ),
    ).to(device)

    if args['load_checkpoint'] != "":
        try:
            load_model(model, f"{args['load_checkpoint']}")
        except:
            print("model not found, initializing randomly")

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    exp = PyTorchExperiment(
        train_dataset=trainSet,
        test_dataset=testSet,
        batch_size=args['batch_size'],
        model=model,
        loss_fn=diffusion_loss(),
        checkpoint_path=model_path,
        experiment_name=args['exp_name'],
        with_wandb=args['wandb'],
        num_workers=torch.get_num_threads() if torch.cuda.is_available() else 0,
        seed=args["seed"],
        args=args,
        save_always=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args['learning_rate'])

    num_epochs = args['epochs']
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=(len(trainSet) * num_epochs),
    )

    exp.train(args['epochs'], optimizer, milestones=args['scheduler'], gamma=args['lr_decay'], scheduler=lr_scheduler)








