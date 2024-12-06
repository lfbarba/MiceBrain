import torch.nn as nn
import torch
import tqdm.auto as tqdm


def flatten(permutation, volume, size=32):
    return volume.permute(permutation).flatten(0, 1).unfold(dimension=1, size=size, step=size).flatten(0, 1)

def unflatten(permutation, flat_volume, shape):
    size = flat_volume.shape[-1]
    permuted_shape = [shape[i] for i in permutation]
    inverse_permutation = [x.item() for x in torch.tensor(permutation).sort().indices]
    unfolded_volume = flat_volume.unflatten(0, (-1, permuted_shape[-2]//size)).transpose(-1, -2).flatten(-3, -2)
    return unfolded_volume.reshape(permuted_shape, 2).permute(inverse_permutation)

class BrainSliceDiffusion(nn.Module):
    def __init__(self, image_shape, channels_indices, unet, buffer=5):
        super(BrainSliceDiffusion, self).__init__()
        self.unet = unet
        self.buffer = buffer
        self.channels = channels_indices

    def inpainting(self, pred_x_0, integer_brain_coord, brain2_slices_rs, reference_image):
        with torch.no_grad():
            gamma = 0.
            pred_x_0[integer_brain_coord[:, 0], integer_brain_coord[:, 1], integer_brain_coord[:, 2], :-1] *= gamma
            pred_x_0[integer_brain_coord[:, 0], integer_brain_coord[:, 1], integer_brain_coord[:, 2], :-1] += (
                                                                                                                          1 - gamma) * brain2_slices_rs[
                                                                                                                                       :,
                                                                                                                                       self.channels].clone().float()
            pred_x_0[:, :, :, -1] *= gamma
            pred_x_0[:, :, :, -1] += (1 - gamma) * reference_image
        return pred_x_0

    def predict_x_0(self, t, x_t, noise_scheduler):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        timesteps = torch.LongTensor([t]).to(device)
        x_0_pred = self.unet(x_t.to(device), timesteps.to(device), return_dict=False)[0]
        return x_0_pred.cpu()

    def guided_diffusion_pipeline(
            self, x_t_start, t_start, t_end, noise_scheduler, num_steps,
            integer_brain_coord, brain2_slices_rs, reference_image,
            batch_size, verbose=True
    ):
        device = x_t_start.device
        x_t = x_t_start.clone()

        timesteps = torch.linspace(t_start, t_end + self.buffer, num_steps + 1).int()
        iterator = tqdm(range(1, len(timesteps)))
        for i in iterator:
            t = timesteps[i - 1]
            target_t = timesteps[i]

            permutations = [
                [0, 1, 2, 3],
                [1, 2, 0, 3],
                [2, 0, 1, 3]
            ]
            with torch.no_grad():
                permutation = permutations[i % len(permutations)]
                iterator.set_postfix({"state": f"rolling"})
                shifts = torch.randint(-16, 16, (3,))
                rolled = x_t.roll([x.item() for x in shifts], dims=[0, 1, 2])
                axial_batches = flatten(permutation, rolled)
                pred_x_0 = torch.zeros_like(axial_batches)
                assert (len(axial_batches) % batch_size == 0)
                for j, x_t_batch in enumerate(
                        axial_batches.unfold(dimension=0, size=batch_size, step=batch_size).permute(0, 3, 1, 2)):
                    pred_x_0[j * batch_size: (j + 1) * batch_size] = self.predict_x_0(t, x_t_batch, noise_scheduler)
                    iterator.set_postfix({"state": f"running batch {j}"})

                iterator.set_postfix({"state": f"starting making contiguous"})
                if not pred_x_0.is_contiguous():
                    pred_x_0 = pred_x_0.contiguous()
                iterator.set_postfix({"state": f"starting unflattening"})
                pred_x_0 = unflatten(permutation, pred_x_0, x_t.shape)
                del axial_batches
                iterator.set_postfix({"state": f"unrolling"})
                pred_x_0 = pred_x_0.roll([-x.item() for x in shifts], dims=[0, 1, 2])

                pred_x_0 = self.inpainting(pred_x_0, integer_brain_coord, brain2_slices_rs, reference_image)

                new_timestep = torch.LongTensor([target_t]).to(device)
                x_t = noise_scheduler.add_noise(pred_x_0, torch.randn_like(pred_x_0), new_timestep)

        return x_t


import torch.nn as nn


class BrainSliceConditionalDiffusion(nn.Module):
    def __init__(self, image_shape, channels_indices, unet, buffer=5):
        super(BrainSliceConditionalDiffusion, self).__init__()
        self.unet = unet
        self.buffer = buffer
        self.channels = channels_indices

    def inpainting(self, pred_x_0, integer_brain_coord, brain2_slices_rs, reference_image):
        with torch.no_grad():
            pred_x_0[integer_brain_coord[:, 0], integer_brain_coord[:, 1], integer_brain_coord[:, 2], :] *= 0
            pred_x_0[integer_brain_coord[:, 0], integer_brain_coord[:, 1], integer_brain_coord[:, 2],
            :] += brain2_slices_rs[:, self.channels].clone().float()
            # pred_x_0[:, :, :, -1] *= 0
            # pred_x_0[:, :, :, -1] += reference_image
        return pred_x_0

    def predict_x_0(self, t, x_t, noise_scheduler):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        timesteps = torch.LongTensor([t]).to(device)
        x_0_pred = self.unet(x_t.to(device), timesteps.to(device), return_dict=False)[0]
        return x_0_pred.cpu()

    def step(self, t, target_t, x_t, noise_scheduler):
        x_0_pred = self.predict_x_0(t, x_t, noise_scheduler)
        new_timestep = torch.LongTensor([target_t]).to(device)
        new_x_t = noise_scheduler.add_noise(x_0_pred, torch.randn_like(x_0_pred), new_timestep).to(device)

        return new_x_t, noise_pred

    def diffusion_pipeline(self, x_t_start, t_start, t_end, noise_scheduler, num_steps=50, verbose=False):
        with torch.no_grad():
            x_t = x_t_start.clone()

            timesteps = torch.linspace(t_start, t_end, num_steps + 1).int()
            for i in tqdm(range(1, len(timesteps)), disable=not verbose):
                t = timesteps[i - 1]
                target_t = timesteps[i]
                x_t, _ = self.step(t, target_t, x_t, noise_scheduler)
            return x_t

    def guided_diffusion_pipeline(
            self, x_0_start, t_start, t_end, noise_scheduler, num_steps,
            integer_brain_coord, brain2_slices_rs, reference_image,
            batch_size, verbose=True
    ):
        pred_x_0 = x_0_start.clone()

        timesteps = torch.linspace(t_start, t_end + self.buffer, num_steps + 1).int()
        iterator = tqdm(range(1, len(timesteps)))
        for i in iterator:
            t = timesteps[i - 1]

            timestep = torch.LongTensor([t]).to(device)
            x_t = noise_scheduler.add_noise(pred_x_0, torch.randn_like(pred_x_0), timestep)

            permutations = [
                [0, 1, 2, 3],
                [1, 2, 0, 3],
                [2, 0, 1, 3]
            ]
            with torch.no_grad():
                permutation = permutations[i % len(permutations)]

                iterator.set_postfix({"state": f"rolling"})
                shifts = torch.randint(-16, 16, (3,))
                rolled = x_t.roll([x.item() for x in shifts], dims=[0, 1, 2])
                rolled_reference_image = reference_image.roll([x.item() for x in shifts], dims=[0, 1, 2])

                axial_batches = flatten(permutation, rolled)
                axial_conditioning = flatten(permutation, rolled_reference_image.unsqueeze(-1))
                pred_x_0 = torch.zeros_like(axial_batches)
                assert (len(axial_batches) % batch_size == 0)

                unfolded_x_t = axial_batches.unfold(dimension=0, size=batch_size, step=batch_size).permute(0, 3, 1, 2)
                unfolded_conditioning = axial_conditioning.unfold(dimension=0, size=batch_size,
                                                                  step=batch_size).permute(0, 3, 1, 2)
                unfolded_input = torch.cat([unfolded_x_t, unfolded_conditioning], dim=2)
                for j, input in enumerate(unfolded_input):
                    pred_x_0[j * batch_size: (j + 1) * batch_size] = self.predict_x_0(t, input, noise_scheduler)
                    iterator.set_postfix({"state": f"running batch {j}"})

                iterator.set_postfix({"state": f"starting making contiguous"})
                if not pred_x_0.is_contiguous():
                    pred_x_0 = pred_x_0.contiguous()

                iterator.set_postfix({"state": f"starting unflattening"})
                pred_x_0 = unflatten(permutation, pred_x_0, x_t.shape)
                del axial_batches, axial_conditioning, unfolded_conditioning, unfolded_input, unfolded_x_t

                iterator.set_postfix({"state": f"unrolling"})
                pred_x_0 = pred_x_0.roll([-x.item() for x in shifts], dims=[0, 1, 2])
                pred_x_0 = self.inpainting(pred_x_0, integer_brain_coord, brain2_slices_rs, reference_image)

        #         x_t = self.diffusion_pipeline(
        #             x_t, t_end + self.buffer, t_end,
        #             noise_scheduler, num_steps=self.buffer,
        #             verbose=verbose
        #         )

        return x_t
