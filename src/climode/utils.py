from typing import Tuple
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import cm, pyplot as plt
import random
import urllib
from numpy import load
import os
import torch.optim as optim
import torch
from tqdm.cli import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline
from scipy import interpolate
import scipy
from torch import Tensor
from torch.utils.data import DataLoader
import matplotlib.patches as patches
import matplotlib.colors as colors
import properscoring as ps


BOUNDARIES = {
    "NorthAmerica": {"lat_range": (15, 65), "lon_range": (220, 300)},  # 8x14
    "SouthAmerica": {"lat_range": (-55, 20), "lon_range": (270, 330)},  # 14x10
    "Europe": {"lat_range": (30, 65), "lon_range": (0, 40)},  # 6x8
    "SouthAsia": {"lat_range": (-15, 45), "lon_range": (25, 110)},  # 10, 14
    "EastAsia": {"lat_range": (5, 65), "lon_range": (70, 150)},  # 10, 12
    "Australia": {"lat_range": (-50, 10), "lon_range": (100, 180)},  # 10x14
}


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set

    # torch.backends.cudnn.deterministic = True # I don't care about determinism on this level.
    # torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_batched(train_times, data_train_final, lev):
    for idx, year in enumerate(train_times):
        data_per_year = data_train_final.sel(time=slice(str(year), str(year))).load()
        data_values = data_per_year[lev].values
        if idx == 0:
            train_data = torch.from_numpy(data_values).reshape(
                -1, 1, 1, data_values.shape[-2], data_values.shape[-1]
            )
            if year % 4 == 0:
                train_data = torch.cat(
                    (train_data[:236], train_data[240:])
                )  # skipping 29 feb in leap year
        else:
            mid_data = torch.from_numpy(data_values).reshape(
                -1, 1, 1, data_values.shape[-2], data_values.shape[-1]
            )
            if year % 4 == 0:
                mid_data = torch.cat(
                    (mid_data[:236], mid_data[240:])
                )  # skipping 29 feb in leap year
            train_data = torch.cat([train_data, mid_data], dim=1)

    return train_data


def get_train_test_data_without_scales_batched(
    data_path, train_time_scale, val_time_scale, test_time_scale, lev, spectral
):
    data = xr.open_mfdataset(data_path, combine="by_coords")
    # data = data.isel(lat=slice(None, None, -1))
    if lev in ["v", "u", "r", "q", "tisr"]:
        data = data.sel(level=500)
    data = data.resample(time="6h").nearest(
        tolerance="1h"
    )  # Setting data to be 6-hour cycles
    data_train = data.sel(time=train_time_scale).load()
    data_val = data.sel(time=val_time_scale).load()
    data_test = data.sel(time=test_time_scale).load()
    data_global = data.sel(time=slice("2006", "2018")).load()

    max_val = data_global.max()[lev].values.tolist()
    min_val = data_global.min()[lev].values.tolist()

    data_train_final = (data_train - min_val) / (max_val - min_val)
    data_val_final = (data_val - min_val) / (max_val - min_val)
    data_test_final = (data_test - min_val) / (max_val - min_val)

    time_vals = data_test_final.time.values
    train_times = [i for i in range(2006, 2016)]
    test_times = [2017, 2018]
    val_times = [2016]

    train_data = get_batched(train_times, data_train_final, lev)
    test_data = get_batched(test_times, data_test_final, lev)
    val_data = get_batched(val_times, data_val_final, lev)

    t = [i for i in range(365 * 4)]
    time_steps = torch.tensor(t).view(-1, 1)
    return (
        train_data,
        val_data,
        test_data,
        time_steps,
        data.lat.values,
        data.lon.values,
        max_val,
        min_val,
        time_vals,
    )


def get_train_test_data_batched_regional(
    data_path, train_time_scale, val_time_scale, test_time_scale, lev, spectral, region
):
    data = xr.open_mfdataset(data_path, combine="by_coords")

    lon_range = slice(
        BOUNDARIES[region]["lon_range"][0], BOUNDARIES[region]["lon_range"][1]
    )
    lat_range = slice(
        BOUNDARIES[region]["lat_range"][0], BOUNDARIES[region]["lat_range"][1]
    )
    data = data.sel(lat=lat_range, lon=lon_range)
    # data = data.isel(lat=slice(None, None, -1))
    if lev in ["v", "u", "r", "q", "tisr"]:
        data = data.sel(level=500)
    data = data.resample(time="6h").nearest(
        tolerance="1h"
    )  # Setting data to be 6-hour cycles
    data_train = data.sel(time=train_time_scale).load()
    data_val = data.sel(time=val_time_scale).load()
    data_test = data.sel(time=test_time_scale).load()

    data_global = data.sel(time=slice("2006", "2018")).load()

    max_val = data_global.max()[lev].values.tolist()
    min_val = data_global.min()[lev].values.tolist()

    # breakpoint()
    mean_global = data_global.mean()
    std_global = data_global.std()

    # data_train_final = (data_train - mean_global)/ std_global
    # data_val_final = (data_val - mean_global)/ std_global
    # data_test_final = (data_test - mean_global)/ std_global

    data_train_final = (data_train - min_val) / (max_val - min_val)
    data_val_final = (data_val - min_val) / (max_val - min_val)
    data_test_final = (data_test - min_val) / (max_val - min_val)

    time_vals = data_test_final.time.values
    train_times = [i for i in range(2006, 2016)]
    test_times = [2017, 2018]
    val_times = [2016]

    train_data = get_batched(train_times, data_train_final, lev)
    test_data = get_batched(test_times, data_test_final, lev)
    val_data = get_batched(val_times, data_val_final, lev)

    t = [i for i in range(365 * 4)]
    time_steps = torch.tensor(t).view(-1, 1)
    return (
        train_data,
        val_data,
        test_data,
        time_steps,
        data.lat.values,
        data.lon.values,
        max_val,
        min_val,
        time_vals,
    )


def nll(mean, std, truth, lat, var_coeff):
    normal_lkl = torch.distributions.normal.Normal(mean, 1e-3 + std)
    lkl = -normal_lkl.log_prob(truth)
    loss_val = lkl.mean() + var_coeff * (std**2).sum()
    # loss_val = torch.mean(lkl,dim=(0,1,3,4))
    return loss_val


def evaluation_rmsd_mm(Pred, Truth, lat, lon, max_vals, min_vals, H, W, levels):
    RMSD_final = []
    RMSD_lat_lon = []
    true_lat_lon = []
    pred_lat_lon = []
    for idx, lev in enumerate(levels):
        true_idx = idx
        das_pred = []
        das_true = []
        pred_spectral = Pred[idx].detach().cpu().numpy()
        true_spectral = Truth[true_idx, :, :].detach().cpu().numpy()

        pred = pred_spectral * (max_vals[idx] - min_vals[idx]) + min_vals[idx]

        das_pred.append(
            xr.DataArray(
                pred.reshape(1, H, W),
                dims=["time", "lat", "lon"],
                coords={"time": [0], "lat": lat, "lon": lon},
                name=lev,
            )
        )
        Pred_xr = xr.merge(das_pred)

        true = true_spectral * (max_vals[idx] - min_vals[idx]) + min_vals[idx]

        das_true.append(
            xr.DataArray(
                true.reshape(1, H, W),
                dims=["time", "lat", "lon"],
                coords={"time": [0], "lat": lat, "lon": lon},
                name=lev,
            )
        )
        True_xr = xr.merge(das_true)
        error = Pred_xr - True_xr
        weights_lat = np.cos(np.deg2rad(error.lat))
        weights_lat /= weights_lat.mean()
        rmse = np.sqrt(((error) ** 2 * weights_lat).mean(dim=["lat", "lon"])).mean(
            dim=["time"]
        )
        lat_lon_rmse = np.sqrt((error) ** 2)
        RMSD_lat_lon.append(lat_lon_rmse[lev].values)
        RMSD_final.append(rmse[lev].values.tolist())

    return RMSD_final


def evaluation_rmsd_mm_region(Pred, Truth, lat, lon, max_vals, min_vals, H, W, levels):
    RMSD_final = []
    RMSD_lat_lon = []
    true_lat_lon = []
    pred_lat_lon = []
    for idx, lev in enumerate(levels):
        true_idx = idx
        das_pred = []
        das_true = []
        pred_spectral = Pred[idx, 1:, 1:].detach().cpu().numpy()
        true_spectral = Truth[true_idx, 1:, 1:].detach().cpu().numpy()

        pred = pred_spectral * (max_vals[idx] - min_vals[idx]) + min_vals[idx]

        das_pred.append(xr.DataArray(pred.reshape(1,H-1,W-1),dims=['time','lat','lon'],coords={'time':[0],'lat':lat[1:],'lon':lon[1:]},name=lev))
        Pred_xr = xr.merge(das_pred)

        das_true.append(xr.DataArray(true.reshape(1,H-1,W-1),dims=['time','lat','lon'],coords={'time':[0],'lat':lat[1:],'lon':lon[1:]},name=lev))
        True_xr = xr.merge(das_true)
        error = Pred_xr - True_xr
        weights_lat = np.cos(np.deg2rad(error.lat))
        weights_lat /= weights_lat.mean()
        rmse = np.sqrt(((error) ** 2 * weights_lat).mean(dim=["lat", "lon"])).mean(
            dim=["time"]
        )
        lat_lon_rmse = np.sqrt((error) ** 2)
        RMSD_lat_lon.append(lat_lon_rmse[lev].values)
        RMSD_final.append(rmse[lev].values.tolist())

    return RMSD_final


def evaluation_acc_mm(Pred, Truth, lat, lon, max_vals, min_vals, H, W, levels, clim):
    ACC_final = []

    for idx, lev in enumerate(levels):
        pred_spectral = Pred[idx].detach().cpu().numpy()
        true_spectral = Truth[idx, :, :].detach().cpu().numpy()
        pred_spectral = pred_spectral - clim[idx]
        true_spectral = true_spectral - clim[idx]

        pred = pred_spectral * (max_vals[idx] - min_vals[idx]) + min_vals[idx]
        true = true_spectral * (max_vals[idx] - min_vals[idx]) + min_vals[idx]

        weights_lat = np.cos(np.deg2rad(lat))
        weights_lat /= weights_lat.mean()
        weights_lat = weights_lat.reshape(len(lat), 1)
        weights_lat = weights_lat.repeat(len(lon), 1)

        pred_prime = pred - np.mean(pred)
        true_prime = true - np.mean(true)

        acc = np.sum(weights_lat * pred_prime * true_prime) / np.sqrt(
            np.sum(weights_lat * pred_prime**2) * np.sum(weights_lat * true_prime**2)
        )
        ACC_final.append(acc)

    return ACC_final


def evaluation_acc_mm_region(
    Pred, Truth, lat, lon, max_vals, min_vals, H, W, levels, clim
):
    ACC_final = []

    for idx, lev in enumerate(levels):
        pred_spectral = Pred[idx, 1:].detach().cpu().numpy()
        true_spectral = Truth[idx, 1:, :].detach().cpu().numpy()
        pred_spectral = pred_spectral - clim[idx, 1:]
        true_spectral = true_spectral - clim[idx, 1:]

        pred = pred_spectral * (max_vals[idx] - min_vals[idx]) + min_vals[idx]
        true = true_spectral * (max_vals[idx] - min_vals[idx]) + min_vals[idx]

        weights_lat = np.cos(np.deg2rad(lat[1:]))
        weights_lat /= weights_lat.mean()
        weights_lat = weights_lat.reshape(len(lat[1:]), 1)
        weights_lat = weights_lat.repeat(len(lon), 1)

        pred_prime = pred - np.mean(pred)
        true_prime = true - np.mean(true)

        acc = np.sum(weights_lat * pred_prime * true_prime) / np.sqrt(
            np.sum(weights_lat * pred_prime**2) * np.sum(weights_lat * true_prime**2)
        )
        ACC_final.append(acc)

    return ACC_final


def evaluation_crps_mm(Pred, Truth, lat, lon, max_vals, min_vals, H, W, levels, Sigma):
    CRPS_final = []

    for idx, lev in enumerate(levels):
        pred_spectral = Pred[idx].detach().cpu().numpy()
        true_spectral = Truth[idx, :, :].detach().cpu().numpy()
        std_spectral = Sigma[idx].detach().cpu().numpy()

        pred = pred_spectral * (max_vals[idx] - min_vals[idx]) + min_vals[idx]
        true = true_spectral * (max_vals[idx] - min_vals[idx]) + min_vals[idx]

        crps = ps.crps_gaussian(true_spectral, mu=pred_spectral, sig=std_spectral)
        CRPS_final.append(crps)

    return CRPS_final


def add_constant_info(path):
    data = xr.open_mfdataset(path, combine="by_coords")
    for idx, var in enumerate(["orography", "lsm"]):
        var_value = torch.from_numpy(data[var].values).view(1, 1, 32, 64)
        if idx == 0:
            final_var = var_value
        else:
            final_var = torch.cat([final_var, var_value], dim=1)

    return (
        final_var,
        torch.from_numpy(data["lat2d"].values),
        torch.from_numpy(data["lon2d"].values),
    )


def add_constant_info_region(path, region, H, W):
    data = xr.open_mfdataset(path, combine="by_coords")
    lon_range = slice(
        BOUNDARIES[region]["lon_range"][0], BOUNDARIES[region]["lon_range"][1]
    )
    lat_range = slice(
        BOUNDARIES[region]["lat_range"][0], BOUNDARIES[region]["lat_range"][1]
    )
    data = data.sel(lat=lat_range, lon=lon_range)
    for idx, var in enumerate(["orography", "lsm"]):
        var_value = torch.from_numpy(data[var].values).view(1, 1, H, W)
        if idx == 0:
            final_var = var_value
        else:
            final_var = torch.cat([final_var, var_value], dim=1)

    return (
        final_var,
        torch.from_numpy(data["lat2d"].values),
        torch.from_numpy(data["lon2d"].values),
    )


def get_delta_u(u_vel, t_steps):
    levels = ["z", "t", "t2m", "u10", "v10", "tisr", "v", "u", "r", "q"]
    t = t_steps.flatten().float() * 6
    title = {
        "z": "Geopotential",
        "v10": "v component of wind at 10m",
        "u10": "u component of wind at 10m",
        "t2m": "Temperature at 2m",
        "t": "Temperature at 850hPa pressure",
    }
    input_u_vel = u_vel.view(u_vel.shape[0], u_vel.shape[1], -1)
    coeffs = natural_cubic_spline_coeffs(t, input_u_vel)
    spline = NaturalCubicSpline(coeffs)
    point = t[-1]
    out = spline.derivative(point).view(
        -1, u_vel.shape[2], u_vel.shape[3], u_vel.shape[4]
    )

    return out


def get_gauss_kernel(shape, lat, lon):
    cwd = os.getcwd()
    rows, columns = shape
    kernel = torch.zeros(shape[0] * shape[1], shape[0] * shape[1])
    print(f"Creating global gaussian kernel of shape {kernel.shape}...")
    pos = []
    for i in range(rows):
        for j in range(columns):
            pos.append([lat[i], lon[j]])

    for i in range(rows * columns):
        for j in range(rows * columns):
            dist = torch.sum((torch.tensor(pos[i]) - torch.tensor(pos[j])) ** 2)
            kernel[i][j] = torch.exp(-dist / (2 * 1 * 1))
            
    print("Calculating kernel inverse...")
    kernel_inv = torch.linalg.inv(kernel).numpy()
    np.save("kernel.npy", kernel_inv)


def get_gauss_kernel_region(shape, lat, lon, region):
    cwd = os.getcwd()
    rows, columns = shape
    kernel = torch.zeros(shape[0] * shape[1], shape[0] * shape[1])
    pos = []
    for i in range(rows):
        for j in range(columns):
            pos.append([lat[i], lon[j]])

    for i in range(rows * columns):
        for j in range(rows * columns):
            dist = torch.sum((torch.tensor(pos[i]) - torch.tensor(pos[j])) ** 2)
            kernel[i][j] = torch.exp(-dist / (2 * 1 * 1))

    kernel_inv = torch.linalg.inv(kernel).numpy()
    np.save(str(cwd) + "/kernel_" + str(region) + ".npy", kernel_inv)

def optimize_vel(num: int, data: Tensor, delta_u: Tensor, 
                 vel_model:nn.Module, kernel:Tensor, 
                 H:int, W:int, steps:int=200) -> Tuple[Tensor]:
    """
    Optimize the initial velocity field for a neural climate model.

    Args:
        num (int): Number of samples or batches.
        data (torch.Tensor): Input data tensor for the model (e.g., climate variables).
        delta_u (torch.Tensor): Target velocity difference (or update) to match.
        vel_model (callable): A model class or function that initializes the velocity model.
        kernel (torch.Tensor): A kernel matrix used for calculating correlations or smoothing.
        H (int): Height of the spatial grid.
        W (int): Width of the spatial grid.
        steps (int, optional): Number of optimization steps. Default is 200.

    Returns:
        final_vx (torch.Tensor): Optimized x-component of velocity.
        final_vy (torch.Tensor): Optimized y-component of velocity.
        loss_step (list): Loss values over optimization steps.
        final_out (torch.Tensor): Model's final output after optimization.
    """
    # Initialize the velocity model
    model = vel_model(num, H, W).to(data.device)
    optimizer = optim.Adam(model.parameters(), lr=2)
    best_loss = float("inf")  # Track the best loss encountered
    loss_step = []  # Record loss values for each step
    loss_fn = nn.MSELoss()  # Mean Squared Error loss

    for step in range(steps):
        optimizer.zero_grad()

        # Forward pass through the velocity model
        out, v_x, v_y = model(data)

        # Reshape and expand tensors for kernel application
        kernel_v_x = v_x.view(num, 5, -1, 1)
        kernel_v_y = v_y.view(num, 5, -1, 1)
        kernel_expand = kernel.expand(num, 5, kernel.shape[0], kernel.shape[1])

        # Compute correlations for regularization
        v_x_kernel = torch.matmul(kernel_v_x.transpose(2, 3), kernel_expand)
        final_x = torch.matmul(v_x_kernel, kernel_v_x).mean()
        v_y_kernel = torch.matmul(kernel_v_y.transpose(2, 3), kernel_expand)
        final_y = torch.matmul(v_y_kernel, kernel_v_y).mean()

        # Compute total loss with a regularization term
        vel_loss = loss_fn(delta_u, out.squeeze(dim=1)) + 1e-7 * (final_x + final_y)
        loss_step.append(vel_loss.item())

        # Save the best results based on loss value
        if vel_loss.item() < best_loss:
            best_loss = vel_loss.item()
            final_vx = v_x
            final_vy = v_y
            final_out = out

        # Backpropagation and optimization step
        vel_loss.backward()
        optimizer.step()

    return final_vx, final_vy, loss_step, final_out
def fit_velocity(
    time_idx,
    time_loader,
    Final_train_data,
    data_loader,
    device,
    num_years,
    paths_to_data,
    scale,
    H,
    W,
    types,
    vel_model,
    kernel,
):
    """
    Fits and saves the optimized velocity field for a climate model.

    Args:
        time_idx (list): List of time indices.
        time_loader (iterable): Data loader providing time-related batches.
        Final_train_data (list): Pre-processed training data, organized by time steps.
        data_loader (iterable): Data loader providing input data batches.
        device (torch.device): Device on which computations are performed (e.g., CPU or GPU).
        num_years (int): Number of years (or temporal segments) in the dataset.
        paths_to_data (list): Paths or identifiers for the dataset variables.
        scale (int): Scaling factor for data.
        H (int): Height of the spatial grid.
        W (int): Width of the spatial grid.
        types (str): Descriptor for the type of velocity field being computed (e.g., training or test).
        vel_model (callable): Model class or function for velocity modeling.
        kernel (torch.Tensor): Kernel matrix for velocity correlation.

    Returns:
        None. The computed velocity field is saved to a file.
    """
    num = 0
    kernel = kernel.to(device)
    cwd = os.getcwd()

    # Iterate through time steps and batches
    for idx_steps, time_steps, batch in tqdm(
        zip(time_idx, time_loader, data_loader), colour="green", total=len(time_idx)
    ):
        # Prepare past time indices and data samples
        pst = [time_steps[0].item() - i for i in range(3)]
        pst.reverse()
        pst_idx = [idx_steps[0].item() - i for i in range(3)]
        pst_idx.reverse()
        past_time = torch.tensor(pst).to(device)

        # Reshape batch data to match model's input format
        data = (
            batch[0]
            .to(device)
            .view(num_years, 1, len(paths_to_data) * (scale + 1), H, W)
        )

        # Retrieve and stack past samples
        past_sample = [
            Final_train_data[j].view(
                num_years, -1, len(paths_to_data) * (scale + 1), H, W
            )
            for j in pst_idx
        ]
        past_sample = torch.stack(past_sample).view(num_years, 3, -1, H, W).to(device)

        # Compute the target velocity change
        delta_u = get_delta_u(past_sample, past_time)

        # Optimize the velocity using the given model and kernel
        v_x, v_y, loss_terms, out = optimize_vel(
            num_years, data, delta_u, vel_model, kernel, H, W
        )

        # Concatenate the velocity fields
        final_v = torch.cat([v_x, v_y], dim=1).unsqueeze(dim=0)
        if num == 0:
            Final_v = final_v
        else:
            Final_v = torch.cat([Final_v, final_v], dim=0)
        num += 1

    # Save the final velocity field to disk
    file_path = os.path.join(cwd, f"{types}_vel.npy")
    if os.path.exists(file_path):
        os.remove(file_path)
    np.save(file_path, Final_v.cpu().detach().numpy())


def load_velocity(types):
    cwd = os.getcwd()
    vel = []
    for file in types:
        vel.append(np.load(str(cwd) + "/" + file + "_vel.npy"))

    return (torch.from_numpy(v) for v in vel)


def get_batched_monthly(train_times, data_train_final, lev):
    for idx, year in enumerate(train_times):
        data_per_year = data_train_final.sel(time=slice(str(year), str(year))).load()
        data_values = data_per_year[lev].values
        t_data = torch.from_numpy(data_values).reshape(
            -1, 1, 1, data_values.shape[-2], data_values.shape[-1]
        )
        if idx == 0:
            train_data = t_data
        else:
            train_data = torch.cat([train_data, t_data], dim=1)

    return train_data


def get_train_test_data_without_scales_batched_monthly(
    data_path, train_time_scale, val_time_scale, test_time_scale, lev, spectral
):
    data = xr.open_mfdataset(data_path, combine="by_coords")
    if lev in ["v", "u", "r", "q", "tisr"]:
        data = data.sel(level=500)
    data = data.resample(time="6h").nearest(
        tolerance="1h"
    )  # Setting data to be 6-hour cycles
    data = data.resample(time="MS").mean()

    data_train = data.sel(time=train_time_scale).load()
    data_val = data.sel(time=val_time_scale).load()
    data_test = data.sel(time=test_time_scale).load()
    data_global = data.sel(time=slice("2006", "2018")).load()

    max_val = data_global.max()[lev].values.tolist()
    min_val = data_global.min()[lev].values.tolist()

    data_train_final = (data_train - min_val) / (max_val - min_val)
    data_val_final = (data_val - min_val) / (max_val - min_val)
    data_test_final = (data_test - min_val) / (max_val - min_val)

    time_vals = data_test_final.time.values
    train_times = [i for i in range(2006,2016)]
    test_times = [2017,2018]
    val_times = [2016]

    train_data = get_batched_monthly(train_times, data_train_final, lev)
    test_data = get_batched_monthly(test_times, data_test_final, lev)
    val_data = get_batched_monthly(val_times, data_val_final, lev)

    t = [i for i in range(12)]
    time_steps = torch.tensor(t).view(-1, 1)

    return (
        train_data,
        val_data,
        test_data,
        time_steps,
        data.lat.values,
        data.lon.values,
        max_val,
        min_val,
        time_vals,
    )
