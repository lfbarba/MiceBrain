# %% [markdown]
"""
Slice dataset

We will create a dataset assuming we already have a dataframe with at least one density column

The result are patches of shape (channels, patch_height, patch_width) and a mask of shape (channels,patch_height, patch_width)

The mask will have 1 if data is missing, Data loader can choose to impute or mask loss
"""

import logging
import math
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from matplotlib.patches import Rectangle
import sys
sys.path.append("/Users/Daniel/git/libra-m/MiceBrain/mice")
from mice.datasets.base_dataset import BaseImageDataset, PairedTransform
from scipy.ndimage import rotate
from tqdm import tqdm


def compute_missing_pixels(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each unique brain_section_label (each image), compute how many pixels are missing.

    Instead of rounding the x_section and y_section values, we assign a unique integer for
    each unique value by mapping them according to their sorted order.

    Parameters:
    df : pandas.DataFrame with at least the following columns:
         - 'x_section'
         - 'y_section'
         - 'brain_section_label'

    Returns:
    A DataFrame with brain_section_label and the corresponding missing_pixels count.
    """
    data = []

    # Group by brain_section_label (each image)
    for label, group in df.groupby('brain_section_label'):
        # Map each unique x_section value to an integer value based on the sorted order
        x_unique = np.sort(group['x_reconstructed'].unique())
        x_mapping = {x: i for i, x in enumerate(x_unique)}

        # Map each unique y_reconstructed value to an integer value based on the sorted order
        y_unique = np.sort(group['y_reconstructed'].unique())
        y_mapping = {y: i for i, y in enumerate(y_unique)}

        # Apply the mappings to the group
        x_mapped = group['x_reconstructed'].map(x_mapping)
        y_mapped = group['y_reconstructed'].map(y_mapping)

        # Since the mapping starts with 0 and is continuous for each unique value,
        # The total possible grid size is given by the number of unique keys
        total_pixels = len(x_unique) * len(y_unique)

        # Count the number of unique pixel positions present using the mapped coordinates
        count_present = len(set(zip(x_mapped, y_mapped)))

        # Missing pixels is the difference between the grid size and present pixels
        missing_pixels = total_pixels - count_present

        data.append({
            "brain_section_label": label,
            "missing_pixels": missing_pixels
        })
    new_dataframe = pd.DataFrame(data)
    return new_dataframe

def create_grid_average_single(
    df: pd.DataFrame, pixel_size: float = 0.025, value_col: str = 'intensity'
) -> pd.DataFrame:
    """
    Creates a regular grid for the data points based on the given pixel size (in millimeters).
    A grid is built from 0 to the ceiling of the maximum x and y coordinates.
    Each data point is assigned to a grid cell using np.digitize.
    Then, for each combination of grid cell and z_section, the function averages the measurement values.

    The DataFrame `df` is expected to contain:
      - 'x_section': x coordinate value in millimeters.
      - 'y_section': y coordinate value in millimeters.
      - 'z_section': an identifier for the image slice.
      - A measurement column (default 'intensity') that will be averaged in each grid cell.

    Returns:
      A DataFrame with the following columns:
        - 'z_section': the image slice identifier.
        - 'x_bin': the bin index for x (starting at 0).
        - 'y_bin': the bin index for y (starting at 0).
        - 'x_center': the center x coordinate of the grid cell.
        - 'y_center': the center y coordinate of the grid cell.
        - 'mean_value': the average measurement value in that grid cell.
        - 'count': the number of data points in that grid cell.
    """
    # Create bins for x and y from 0 to the ceiling of the maximum coordinate (inclusive)
    max_x = df['x_section'].max()
    max_y = df['y_section'].max()
    bins_x = np.arange(0, np.ceil(max_x) + pixel_size, pixel_size)
    bins_y = np.arange(0, np.ceil(max_y) + pixel_size, pixel_size)

    # Use np.digitize to map each x_section and y_section to their respective bins.
    # np.digitize returns indices starting at 1, so subtract 1 to have bins start at 0.
    df['x_bin'] = np.digitize(df['x_section'], bins=bins_x) - 1
    df['y_bin'] = np.digitize(df['y_section'], bins=bins_y) - 1

    # Compute the center of each grid cell based on the bin index.
    df['x_center'] = df['x_bin'] * pixel_size + (pixel_size / 2)
    df['y_center'] = df['y_bin'] * pixel_size + (pixel_size / 2)
    # Group by z_section and the x, y bin indices and compute the average of the measurement column.
    agg_df = df.groupby(['z_section', 'x_bin', 'y_bin']).agg(
        mean_value=(value_col, 'mean'),
        count=(value_col, 'size')
    ).reset_index()

    # Calculate grid cell centers for the aggregated data (optional, as they are already unique per bin).
    agg_df['x_center'] = agg_df['x_bin'] * pixel_size + (pixel_size / 2)
    agg_df['y_center'] = agg_df['y_bin'] * pixel_size + (pixel_size / 2)

    return agg_df

def create_grid_average(df: pd.DataFrame, pixel_size: float, value_cols: list) -> pd.DataFrame:
    """
    Creates a regular grid for the data points using a grid that spans from 0 to the ceiling of the
    maximum x and y coordinates with bins at intervals of pixel_size (in millimeters). For each grid cell,
    averages are computed for each column in value_cols and the number of data points in each cell is recorded.

    Parameters:
      df         : Input DataFrame. Expected to contain at least 'x_section', 'y_section', and 'z_section'
                   plus hundreds of measurement columns.
      pixel_size : The size of each grid cell in millimeters (e.g. 0.025 mm for 25 micrometers).
      value_cols : List of column names (strings) that contain measurement values to be averaged.

    Returns:
      A DataFrame with:
         - 'z_section': the image slice identifier.
         - 'x_bin'    : x bin index.
         - 'y_bin'    : y bin index.
         - For each column in value_cols, the mean value in that grid cell.
         - 'count'    : Number of data points in that grid cell.
         - 'x_center' : The center x coordinate of the grid cell.
         - 'y_center' : The center y coordinate of the grid cell.
    """
    # Build bins for x and y starting at 0 up to the ceiling of the maximum coordinate plus pixel_size
    max_x = df['x_section'].max()
    max_y = df['y_section'].max()
    bins_x = np.arange(0, np.ceil(max_x) + pixel_size, pixel_size)
    bins_y = np.arange(0, np.ceil(max_y) + pixel_size, pixel_size)

    # Use np.digitize to assign each coordinate to a bin.
    # np.digitize returns indices starting at 1, so subtract 1 for 0-based indexing.
    df['x_bin'] = np.digitize(df['x_section'], bins=bins_x) - 1
    df['y_bin'] = np.digitize(df['y_section'], bins=bins_y) - 1

    # Calculate the center of each grid cell.
    df['x_center'] = df['x_bin'] * pixel_size + (pixel_size / 2)
    df['y_center'] = df['y_bin'] * pixel_size + (pixel_size / 2)

    # For memory efficiency, convert grouping keys to more efficient dtypes.
    df['z_section'] = df['z_section'].astype('category')
    df['x_bin'] = df['x_bin'].astype('int32')
    df['y_bin'] = df['y_bin'].astype('int32')

    group_cols = ['z_section', 'x_bin', 'y_bin']

    # Compute the mean for each measurement column from value_cols for every grid cell.
    # Instead of using a single aggregation dictionary that computes count from one column,
    # we first compute the mean and then separately add in the count from group sizes.
    agg_means = df.groupby(group_cols, observed=True, sort=False).agg({col: 'mean' for col in value_cols}).reset_index()

    # Compute the count (number of points) in each group.
    counts = df.groupby(group_cols, observed=True, sort=False).size().reset_index(name='count')

    # Merge the aggregate mean values and counts on the grouping columns.
    agg_df = pd.merge(agg_means, counts, on=group_cols)

    # Recalculate grid cell center positions for the aggregated results (x_center and y_center).
    agg_df['x_center'] = agg_df['x_bin'] * pixel_size + (pixel_size / 2)
    agg_df['y_center'] = agg_df['y_bin'] * pixel_size + (pixel_size / 2)

    return agg_df

def plot_sections(df: pd.DataFrame, value_col: str = 'density', ncols: int = 4):
    """
    Loads grid-averaged image data from a CSV file and plots each image section using imshow.
    Each section (identified by 'z_section') is displayed as a panel in a figure containing ncols columns.

    Parameters:
      data_file : str
          Path to the CSV file containing the grid-averaged data. The CSV file is expected to include
          the following columns:
              - 'z_section' : Identifier for each image section.
              - 'x_bin'     : The x-index of the grid cell.
              - 'y_bin'     : The y-index of the grid cell.
              - value_col   : The column of measurement values to plot. Default is 'mean_value'.
      value_col : str, optional
          Name of the measurement column to use for plotting. Default is 'mean_value'.
      ncols     : int, optional
          Number of columns of subplots in the output figure. Default is 4.
    """
    # Load data from CSV

    # Get the unique sections (assumes 'z_section' identifies each image)
    sections = sorted(df['z_section'].unique())
    n_sections = len(sections)
    nrows = math.ceil(n_sections / ncols)

    # Create a figure with a grid for subplots.
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 4 * nrows))
    # Flatten axes in case it's a 2D array
    if n_sections > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for i, section in enumerate(sections):
        ax = axes[i]
        # Filter the DataFrame to just include data for the current section.
        section_df = df[df['z_section'] == section]

        # Pivot the DataFrame to form a 2D grid using x_bin and y_bin.
        # Rows (y-axis) and columns (x-axis) are based on the grid indices.
        grid = section_df.pivot(index='y_bin', columns='x_bin', values=value_col)

        # If the pivot table is empty or incomplete, fill missing values with NaN.
        grid = grid.sort_index(ascending=True)
        grid = grid.reindex(sorted(grid.columns), axis=1)

        # Display the grid using imshow.
        im = ax.imshow(grid.values, aspect='auto', interpolation='none', origin='upper')
        ax.set_title(f'Section: {section}')
        ax.set_xlabel('x_bin')
        ax.set_ylabel('y_bin')

        # Add a colorbar to each subplot.
        fig.colorbar(im, ax=ax)

    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def extract_patch_single(image: np.ndarray, start_row: int, start_col: int, patch_height: int, patch_width: int) -> np.ndarray:
    """
    Extracts a patch of size (patch_height, patch_width) from the given 2D image starting at (start_row, start_col).
    If the patch extends beyond the image boundaries, the missing values are filled with NaN.
    """
    img_rows, img_cols = image.shape
    patch = np.full((patch_height, patch_width), np.nan, dtype=image.dtype)

    end_row = min(start_row + patch_height, img_rows)
    end_col = min(start_col + patch_width, img_cols)

    available_rows = end_row - start_row
    available_cols = end_col - start_col

    patch[:available_rows, :available_cols] = image[start_row:end_row, start_col:end_col]
    return patch

def sample_rotated_patches(
    df: pd.DataFrame,
    patch_width: int,
    patch_height: int,
    num_patches: int,
    value_col: str,
    rotation_range: tuple = (-45, 45),
    save_metadata=True
) -> dict:
    """
    For each unique section (identified by 'z_bin') in the DataFrame, this function extracts a number of random patches
    by first rotating the entire image and then sampling a patch from the rotated image.

    For each patch:
      1. The grid image is constructed by pivoting the DataFrame using 'y_bin' as rows and 'x_bin' as columns.
      2. A random angle is chosen from the provided rotation_range.
      3. The entire grid image is rotated by that angle using reshape=True.
      4. A random starting coordinate is chosen such that a patch of size (patch_height, patch_width) is extracted
         from the rotated image. If the patch falls partly outside the image, missing values are filled with NaN.

    Parameters:
      df           : DataFrame with columns 'x_bin', 'y_bin', 'z_bin', and a measurement column (value_col).
      patch_width  : Width (number of columns) of the extracted patch.
      patch_height : Height (number of rows) of the extracted patch.
      num_patches  : Number of patches to extract per section.
      value_col    : The column name in df used to form the image grid.
      rotation_range: Tuple (min_angle, max_angle) to randomly choose rotation angles. Default is (-45, 45).

    Returns:
      A dictionary where keys are unique z_bin identifiers and values are lists of dictionaries.
      Each dictionary contains:
          'patch': The extracted patch (numpy array of shape (patch_height, patch_width)).
          'start_row': The starting row coordinate in the rotated image.
          'start_col': The starting column coordinate in the rotated image.
          'rotated_image': The full rotated image (2D numpy array).
          'angle': The rotation angle applied to the original image.
    """
    patches_by_section = {}

    for z in df['z_section'].unique():
        # Filter rows corresponding to the section.
        df_section = df[df['z_section'] == z]

        # Create the full grid image.
        grid_df = df_section.pivot(index='y_bin', columns='x_bin', values=value_col)
        grid_df = grid_df.reindex(
            index=range(int(df_section['y_bin'].min()), int(df_section['y_bin'].max()) + 1),
            columns=range(int(df_section['x_bin'].min()), int(df_section['x_bin'].max()) + 1)
        )
        grid = grid_df.values.astype(np.float64)

        patch_info_list = []
        for j in range(num_patches):
            # Choose a random rotation angle.
            angle = np.random.uniform(*rotation_range)
            # Rotate the entire image with reshape=True so that the rotated image includes all data.
            rotated_image = rotate(grid, angle=angle, reshape=True, order=1, mode='constant', cval=np.nan)
            img_rows, img_cols = rotated_image.shape

            # Select a random starting coordinate for the patch.
            # Ensure we choose a start coordinate such that at least one pixel exists.
            max_start_row = img_rows - patch_height + 1 if img_rows >= patch_height else 1
            max_start_col = img_cols - patch_width + 1 if img_cols >= patch_width else 1
            start_row = np.random.randint(0, max_start_row)
            start_col = np.random.randint(0, max_start_col)

            patch_ = extract_patch(rotated_image, start_row, start_col, patch_height, patch_width)
            patch_name = f"section_{z}-patch_{j}-r_{start_row}-c{start_col}-a_{angle}"
            mask_name = f"section_{z}-mask_{j}-r_{start_row}-c{start_col}-a_{angle}"
            mask = patch_ == np.nan
            patch_ = torch.tensor(patch_)
            mask = torch.tensor(mask)
            torch.save(patch_, f"/Users/Daniel/mlibra-data/merfish/patches/{patch_name}.pt")
            torch.save(mask, f"/Users/Daniel/mlibra-data/merfish/patches/{mask_name}.pt")
            if save_metadata:

                patch_info = {
                    'patch': patch,
                    'start_row': start_row,
                    'start_col': start_col,
                    'rotated_image': rotated_image,
                    'angle': angle
                }
                patch_info_list.append(patch_info)
        if save_metadata:
            patches_by_section[z] = patch_info_list
    return patches_by_section

def plot_patches_with_overlay(patches_info: list, ncols: int = 4):
    """
    For each patch, this function plots the rotated full image with a red rectangle indicating the patch location.

    Parameters:
      patches_info : List of dictionaries as produced by sample_rotated_patches for a section.
      ncols        : Number of subplot columns in the figure.
    """
    num_patches = len(patches_info)
    nrows = int(np.ceil(num_patches / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows))

    if num_patches > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    idx=0
    for keys in patches_info.keys():

        ax = axes[idx]
        info = patches_info[keys][0]
        rotated_image = info['rotated_image']
        start_row = info['start_row']
        start_col = info['start_col']
        patch = info['patch']
        angle = info['angle']
        img_rows, img_cols = rotated_image.shape[0: 2]

        im = ax.imshow(rotated_image, aspect='auto', interpolation='none', origin='upper', cmap='viridis',alpha=0.1)
        ax.set_title(f"Patch {idx+1}\nAngle: {angle:.1f}°")
        ax.set_xlabel("x coordinate")
        ax.set_ylabel("y coordinate")

        # Draw red rectangle showing patch region.
        rect = Rectangle((start_col, start_row), patch.shape[1], patch.shape[0],
                         linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        fig.colorbar(im, ax=ax)
        idx += 1

    # Remove any unused axes.
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def create_multichannel_grid(df_section: pd.DataFrame, channels: list) -> np.ndarray:
    """
    Creates a multi-channel image by pivoting for each channel.
    Each channel is pivoted using 'y_bin' (rows) and 'x_bin' (columns) and then the results are
    stacked to form a 3D array of shape (height, width, num_channels).
    """
    # Determine the full range of indices.
    y_min = int(df_section['y_bin'].min())
    y_max = int(df_section['y_bin'].max())
    x_min = int(df_section['x_bin'].min())
    x_max = int(df_section['x_bin'].max())
    row_index = range(y_min, y_max + 1)
    col_index = range(x_min, x_max + 1)

    channel_grids = []
    for channel in channels:
        grid_df = df_section.pivot(index='y_bin', columns='x_bin', values=channel)
        grid_df = grid_df.reindex(index=row_index, columns=col_index)
        channel_grids.append(grid_df.values.astype(np.float64))
    # Stack the channel grids along the third axis.
    multichannel_image = np.dstack(channel_grids)
    return multichannel_image

def extract_patch(image: torch.Tensor, start_row: int, start_col: int, patch_height: int, patch_width: int) -> torch.Tensor:
    """
    Extracts a patch from the image starting at (start_row, start_col).
    Works for both single-channel (2D) and multi-channel (3D) images.
    If the patch extends beyond image boundaries, missing values are filled with NaN.
    """
    patch = torch.Tensor()
    if image.ndim == 2:
        img_rows, img_cols = image.shape
        patch = torch.full((patch_height, patch_width), float('nan'), dtype=image.dtype, device=image.device)
        end_row = min(start_row + patch_height, img_rows)
        end_col = min(start_col + patch_width, img_cols)
        patch[0:end_row - start_row, 0:end_col - start_col] = image[start_row:end_row, start_col:end_col]
    elif image.ndim == 3:
        img_rows, img_cols, channels = image.shape
        patch = torch.full((patch_height, patch_width, channels), float('nan'), dtype=image.dtype, device=image.device)
        end_row = min(start_row + patch_height, img_rows)
        end_col = min(start_col + patch_width, img_cols)
        patch[0:end_row - start_row, 0:end_col - start_col, :] = image[start_row:end_row, start_col:end_col, :]
    else:
        raise ValueError("Unsupported image dimensions.")
    return patch

def rotate_image_torch(image: torch.Tensor, angle: float) -> torch.Tensor:
    """
    Rotates an image (either 2D or 3D with channels last) by a given angle in degrees.
    The output image is reshaped to fully contain the rotated image.

    Parameters:
      image: torch.Tensor of shape (H, W) or (H, W, C)
      angle: rotation angle in degrees

    Returns:
      Rotated image as a torch.Tensor with the same channel convention as input.
      Areas outside the boundaries are assigned NaN.
    """
    # Convert angle to radians.
    rad = math.radians(angle)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)

    # Convert image tensor to shape [N, C, H, W]
    # If input is 2D, add a channel dimension.
    if image.ndim == 2:
        image = image.unsqueeze(0).unsqueeze(0)  # shape [1, 1, H, W]
    elif image.ndim == 3:
        # Assume channels last; rearrange to channels-first.
        image = image.permute(2, 0, 1).unsqueeze(0)  # shape [1, C, H, W]
    else:
        raise ValueError("Unsupported image dimensions. Must be 2D or 3D with channels last.")

    _, C, H, W = image.shape

    # Compute new dimensions to ensure the entire rotated image fits.
    new_H = int(math.ceil(abs(H * cos_a) + abs(W * sin_a)))
    new_W = int(math.ceil(abs(W * cos_a) + abs(H * sin_a)))

    # Compute centers of original and new images.
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    ncx, ncy = (new_W - 1) / 2.0, (new_H - 1) / 2.0

    # Compute the offsets needed.
    offset_x = ncx - (cos_a * cx - sin_a * cy)
    offset_y = ncy - (sin_a * cx + cos_a * cy)

    # Construct affine matrix. Devices are matched with the image.
    theta = torch.tensor([[cos_a, -sin_a, offset_x],
                          [sin_a,  cos_a, offset_y]], dtype=torch.float32, device=image.device)
    theta = theta.unsqueeze(0)  # shape [1, 2, 3]

    # Generate grid and sample.
    grid = F.affine_grid(theta, size=[1, C, new_H, new_W], align_corners=False)
    rotated = F.grid_sample(image, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

    # Identify areas that were sampled from out-of-bound coordinates.
    ones = torch.ones_like(image)
    mask = F.grid_sample(ones, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    rotated[mask < 0.999] = float('nan')

    # Convert back to original dimensions.
    if rotated.shape[1] == 1:
        rotated = rotated.squeeze(0).squeeze(0)  # shape [H, W]
    else:
        rotated = rotated.squeeze(0).permute(1, 2, 0)  # shape [H, W, C]

    return rotated


def sample_random_patches(out_path, df: pd.DataFrame, patch_width: int, patch_height: int, num_patches: int,
                          channels, device: torch.device, rotation_range: tuple = (-45, 45), save_metadata=False) -> dict:
    """
    For each unique section in 'z_section', creates a multi-channel image by pivoting via the provided channels,
    rotates the image by a random angle (from rotation_range) using PyTorch for acceleration,
    and then extracts a patch (which includes all channels) from the rotated image.

    All tensors are allocated on the given device.

    Parameters:
      df           : DataFrame with columns 'x_bin', 'y_bin', 'z_section' and measurement columns.
      patch_width  : Width (number of columns) of the desired patch.
      patch_height : Height (number of rows) of the desired patch.
      num_patches  : Number of patches to extract per section.
      channels     : List of column names to use for creating the multi-channel image.
      device       : torch.device to which all tensors are sent.
      rotation_range: Tuple (min_angle, max_angle) for choosing a random rotation angle.
      save_metadata: If True, returns metadata for each patch.

    Returns:
      A dictionary where keys are unique z_section identifiers and values are lists of dictionaries.
      Each dictionary contains:
         'patch': Extracted patch as a torch.Tensor.
         'start_row': Starting row in the rotated image.
         'start_col': Starting column in the rotated image.
         'rotated_image': The full rotated image as a torch.Tensor.
         'angle': Rotation angle applied.
    """
    patches_by_section = {}

    for i,z in enumerate(df['z_section'].unique()):
        df_section = df[df['z_section'] == z]
        # Create multi-channel image using pivot (all channels together).
        logging.info(f"Creating multi-channel grid for section {i} of {len(df['z_section'].unique())}")
        image_np = create_multichannel_grid(df_section, channels)
        # Convert to torch tensor with float32 type on the given device.
        logging.debug(f"Converting to torch tensor on device {device}")
        image_torch = torch.tensor(image_np, dtype=torch.float32, device=device)

        patch_info_list = []
        for j in tqdm(range(num_patches)):
            logging.debug(f"Extracting patch {j} for section {z}")
            angle = float(np.random.uniform(*rotation_range))
            # Rotate the image using the PyTorch accelerated function.
            logging.debug(f"Rotating image by {angle:.2f} degrees")
            rotated_image = rotate_image_torch(image_torch, angle)
            img_rows, img_cols = rotated_image.shape[:2]

            max_start_row = img_rows - patch_height + 1 if img_rows >= patch_height else 1
            max_start_col = img_cols - patch_width + 1 if img_cols >= patch_width else 1
            start_row = np.random.randint(0, max_start_row)
            start_col = np.random.randint(0, max_start_col)

            logging.debug(f"Extracting patch at row {start_row}, col {start_col}")
            patch_ = extract_patch(rotated_image, start_row, start_col, patch_height, patch_width)

            if patch_.isnan().all():
                pass


            patch_ = patch_.permute(2, 0, 1)
            assert patch_.shape == (len(channels), patch_height, patch_width)
            patch_name = f"section_{z}-patch_{j}-r_{start_row}-c_{start_col}-a_{angle:.2f}"
            mask_name = f"section_{z}-mask_{j}-r_{start_row}-c_{start_col}-a_{angle:.2f}"

            # Create a mask indicating where the patch is NaN.
            mask = torch.isnan(patch_)
            # Save the patch and mask.
            logging.debug(f"Saving patch and mask to {out_path}")
            torch.save(patch_, (out_path / f"{patch_name}.pt"))
            logging.debug(f"Saving mask to {out_path}")
            torch.save(mask, (out_path / f"{mask_name}.pt"))

            if save_metadata:
                logging.debug(f"Saving metadata for patch {j}")
                patch_info = {
                    'patch': patch_,
                    'start_row': start_row,
                    'start_col': start_col,
                    'rotated_image': rotated_image,
                    'angle': angle
                }
                patch_info_list.append(patch_info)
        if save_metadata:
            patches_by_section[z] = patch_info_list
    return patches_by_section



# %% [markdown]
"""
Class that loads a dataset of patch files and their corresponding masks.
"""
class PatchFileDataset(BaseImageDataset):

    def __init__(self, path, im_size, lr_forward_function=lambda x:x,
                 rescale=None, clip_range=None, normalize_range=False, rotation_angle=None, num_defects=None,
                 contrast=None, train_transform=False, crop=None, gray_background=False,
                 to_synthetic=False, means=None, stds=None, channels=None):
        super().__init__(path, lr_forward_function=lr_forward_function,
                 rescale=rescale, clip_range=clip_range, normalize_range=normalize_range, rotation_angle=rotation_angle, num_defects=num_defects,
                 contrast=contrast, train_transform=train_transform, crop=crop,
                         to_synthetic=to_synthetic)
        if type(path) == str:
            self.path = Path(path)
        else:
            self.path = path
        self.im_size = im_size
        if channels is not None:
            self.channels = channels
        # search for all files with "patch" in  path
        self.images = [p.name for p in Path(path).glob("*patch*.pt")]
        # masks and patch have same name except for the keyword "mask" or "patch"
        # build a dictionary to match them
        # self.masks = { p: Path(str(p).replace("patch", "mask")) for p in self.images}
        # zip the images and masks
        self.available_channels=torch.load(self.path / self.images[0]).shape[0]
        if channels is not None:
            self.channels = channels
            self.available_channels = len(channels)
        else:
            self.channels = list(range(self.available_channels))
        self.means=None
        self.stds=None
        if means is not None:
            self.means = means[channels]
        if stds is not None:
            self.stds = stds[channels]
        print("loading imagges")
        #self.stacked_img = [torch.load(self.path / p) for p in self.images]
        #self.masked_img = [torch.load(self.path / p) for p in self.masks]

        self.lr_forward_function = lr_forward_function

    def __getitem__(self, idx):
        stacked_img = torch.load(self.path / self.images[idx])
        #mask = torch.load(self.path / self.masks[self.images[idx]])
        #stacked_img = self.stacked_img[idx]
        stacked_img = stacked_img[self.channels]
        #mask = self.masked_img[idx]
        #mask = mask[self.channels]
        #mask = mask[self.channels]
        if self.means is not None:
            return (stacked_img - self.means[None, :, None]) / self.stds[None, :, None]
        else:
            # I created the dataset with the assumption that the images are channel, height, width
            # but the data loader seems to assume  width, channel, height
            # so I permute the tensor to match the data loader
            stacked_img=stacked_img.permute(1, 0, 2)
            # for experiment I will replace nan in stacked_img with 0
            stacked_img[stacked_img.isnan()] = 0
            return stacked_img

    def __len__(self):
        return len(self.images)




parser = ArgumentParser()
parser.add_argument("--data-file", type=str, required=False, help="Path to the parquet file containing the data.")
parser.add_argument("--pixel-size", type=float, default=0.025, help="Size of each grid cell in millimeters.")
parser.add_argument("--value-cols", type=str, nargs='+', required=False, help="List of columns to average in each grid cell.")
parser.add_argument("--patch-width", type=int, default=128, help="Width of the extracted patch.")
parser.add_argument("--patch-height", type=int, default=1, help="Height of the extracted patch.")
parser.add_argument("--num-patches", type=int, default=2, help="Number of patches to extract per section.")
parser.add_argument("--channels", type=str, nargs='+', required=False, help="List of channels to use for multi-channel image.")
parser.add_argument("--rotation-range", type=float, nargs=2, default=(-45, 45), help="Range of rotation angles.")
parser.add_argument("--save-metadata", action="store_true", help="Save metadata for each patch.")
parser.add_argument("--aggregated-file", type=str, required=False, help="Path to the parquet file containing the aggregated data.")
parser.add_argument("--environment", type=str, required=False, help="Environment to run the script in.")
parser.add_argument("--device", type=str, default="cpu", help="Device to use for tensor operations.")
parser.add_argument("--test", action="store_true", help="Run the script with test arguments.")
parser.add_argument("--logging-level", type=str, default="INFO", help="Logging level for the script.")

test_args = [
    "--data-file", "combined_cell_filtered_w500genes_density_minimal_train.parquet",
    "--aggregated-file", "combined_cell_filtered_w500genes_density_minimal_train_aggregated.parquet",
    "--pixel-size", "0.025",
    "--value-cols", "density",
    "--patch-width", "128",
    "--patch-height", "1",
    "--num-patches", "1000",
    "--channels", "density",
    "--rotation-range", "-45", "45",
    "--device", "cpu",
    "--logging-level", "INFO"
]

if __name__ == "__main__":
    print("Begin")
    args = parser.parse_args()
    if args.test:
        args = parser.parse_args(test_args)
        args.environment = "local"

    if args.environment == "local":
        merfish_path = "/Users/Daniel/mlibra-data/merfish/"
    if args.environment == "runai":
        merfish_path = "/s3/mlibra/mlibra-data/merfish/"
    if args.logging_level == "DEBUG":
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    logging.info("Starting script")
    merfish_path = Path(merfish_path)
    merfish_out_path = merfish_path / "patches"
    merfish_out_path.mkdir(exist_ok=True)
    patch_width = args.patch_width
    patch_height = args.patch_height
    num_patches = args.num_patches
    dataset_name = f"merfish_{patch_width}x{patch_height}"
    dataset_path = merfish_out_path / dataset_name
    dataset_path.mkdir(exist_ok=True, parents=True)
    device = torch.device(args.device)

    merfish_data = merfish_path / args.data_file
    agg_data = merfish_path / args.aggregated_file
    logging.info(f"Loading data from {merfish_data}")
    if agg_data.exists():
        logging.info(f"Loading aggregated data from {agg_data}")
        agg_df = pd.read_parquet(agg_data)
        logging.debug(f"Aggregated data loaded with columns: {agg_df.columns}")
        logging.debug(f"Aggregated data shape: {agg_df.shape}")
       
    else:
        logging.info(f"Aggregated data not found. Creating from {merfish_data}")
        schema = pq.read_table(merfish_data).schema
        gene_cols = [col for col in schema.names if col.startswith("ENS")]
        location_cols = ['x_section', 'y_section', 'z_section']
        df = pd.read_parquet(merfish_data, columns= location_cols + gene_cols + ['density'])
        agg_df = create_grid_average(df, args.pixel_size, gene_cols+['density'])
        logging.info(f"Saving aggregated data")
        agg_df.to_parquet(agg_data)
        del df
    schema = pq.read_table(agg_data).schema
    gen_cols = [col for col in schema.names if col.startswith("ENS")]
    logging.info(f"Creating patches from {dataset_path}")
    sample_random_patches(out_path = dataset_path,
                          df=agg_df,
                          device=device,
                          patch_width=patch_width,
                          patch_height=patch_height,
                          num_patches=num_patches,channels= gen_cols + ['density'])
