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
from pathlib import Path
import math
import sys
#from datasets.slice_dataset import SliceDataset
from mice.datasets.slice_frame_dataset import PatchFileDataset
import numpy as np

from torch.utils.data import Dataset

def get_dataset(kwargs):
    # dataset = SliceDataset(**kwargs)
    dataset = PatchFileDataset(**kwargs)
    torch.manual_seed(0)
    perm = torch.randperm(len(dataset))
    trainSet = torch.utils.data.Subset(dataset, perm[:round(0.95 * len(dataset))])
    testSet = torch.utils.data.Subset(dataset, perm[round(0.95 * len(dataset)):])
    return trainSet, testSet

class diffusion_loss(BaseLoss):

    def __init__(self, mask_loss=False):
        stats_names = ["loss"]
        super(diffusion_loss, self).__init__(stats_names)
        self.mask_loss=mask_loss


    def compute_loss(self, instance, model):
        mse = torch.nn.MSELoss()
        # this considers pixels of height 1 and width 128
        x = instance.flatten(0, 1) # flatten the height pixel from the data.
        # For experiments, nan become zero.
        mask = x.isnan()
        x[mask] = 0

        # TODO: maybe move to the data loader
        target_channel = 145
        condition_channel = -1
        x_0 = x[:, target_channel].to(device)
        # let's use the mask
        mask = mask[:, target_channel].to(device)
        conditioning = x[:, condition_channel].to(device)
        # x_0 = x[:, :-1].to(device)
        # conditioning = x[:, -1:].to(device)

        noise = torch.randn_like(x_0).to(device)
        bs = x_0.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bs,), device=x_0.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        x_t = noise_scheduler.add_noise(x_0, noise, timesteps)
        input = torch.stack([x_t, conditioning], dim=1)
        # input = torch.cat([x_t, conditioning], dim=1)


        model.zero_grad()
        x_0_pred = model(input, timesteps, return_dict=False)[0]

        # fading_factor = noise_scheduler.add_noise(torch.ones(1), torch.zeros(1), timesteps).to(device)
        # noise_factor = noise_scheduler.add_noise(torch.zeros(1), torch.ones(1), timesteps).to(device)
        # x_0_pred = (x_t.unsqueeze(1) - noise_factor[:, None, None] * noise_pred) / fading_factor[:, None, None]
        # x_0_pred = x_0_pred.clip(-3, 3)
        # loss = mse(noise_pred.squeeze(1), noise)
        x_0_pred = x_0_pred.squeeze(1)
        loss = mse(x_0_pred[~mask], x_0[~mask])
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
    parser.add_argument("--model_path", default='', type=str, help="name of models in folder checkpoints to load")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset file or folder")
    parser.add_argument("--im_size", type=int, default=512, help="In the case of tiff, the size of the crops to split the tiff files")
    parser.add_argument("--rescale", type=int, default=512, help="The side length of the images in the dataset")
    parser.add_argument("--means-file", type=str, default="means.npy", help="The file containing the means of the dataset")
    parser.add_argument("--stds-file", type=str, default="stds.npy", help="The file containing the stds of the dataset")
    parser.add_argument("--num_workers", type=int, default=-1, help="Number of workers for the dataloader")


    args = vars(parser.parse_args())
    temp = args["scheduler"].replace(" ", "").replace("[", "").replace("]", "").split(",")
    args["scheduler"] = [int(x) for x in temp]
    args["seed"] = random.randint(0, 20000) if args["seed"] == -1 else args["seed"]
    means_file = args["means_file"]
    stds_file = args["stds_file"]
    means = torch.tensor(np.load(means_file))
    stds = torch.tensor(np.load(stds_file))
    kwargs = {
        "path": args['dataset_path'],
        'im_size':args['im_size'],
        "train_transform": True,
        'stds': stds,
        'means': means
    }
    # NOTE the only argument of interest at the moment is path. The rest are hardcoded on patch creation.
    trainSet, testSet = get_dataset(kwargs)


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps')
    # device = torch.device('cpu')
    print(device)
    model_path = Path(args['model_path'])
    model_path = model_path / "checkpoints"/ f"{args['exp_name']}.pt"
    model_path.parent
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(args['im_size'])
    model = UNet1DModel(
        sample_size=args['im_size'],  # Adjusted to im_size
        in_channels=2,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(32, 64, 128, 256),  # Reduced the number of output channels per block
        down_block_types=(
            "DownBlock1D",
            "DownBlock1D",  # a regular ResNet downsampling block
            "AttnDownBlock1D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock1D",
        ),
        up_block_types=(
            "UpBlock1D",
            "AttnUpBlock1D",  # Keep the same
            "UpBlock1D",
            "UpBlock1D",  # Reduced the number of upsampling blocks accordingly
        ),
    ).to(device)

    if args['load_checkpoint'] != "":
        try:
            load_model(model, f"{args['load_checkpoint']}")
        except:
            print("model not found, initializing randomly")

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    if torch.cuda.is_available():
        num_workers = torch.get_num_threads()
        if args["num_workers"] != -1:
            num_workers = min(num_workers,args["num_workers"])
    else:
        num_workers = 0
    print(f"num_workers: {num_workers}")
    exp = PyTorchExperiment(
        train_dataset=trainSet,
        test_dataset=testSet,
        batch_size=args['batch_size'],
        model=model,
        loss_fn=diffusion_loss(),
        checkpoint_path=model_path,
        experiment_name=args['exp_name'],
        with_wandb=args['wandb'],
        num_workers=num_workers,
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








