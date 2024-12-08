"""
This is an executable file, that contains everything you need.

Use it as python train.py
"""

from tqdm.cli import tqdm
import numpy as np
from torchdiffeq import odeint
import torch
from pathlib import Path
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim
import xarray as xr

import fire

from climode.utils import fit_velocity, get_gauss_kernel, get_batched
from climode.model_function import Optim_velocity, Climate_encoder_free_uncertain


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


def load_velocity(types):
    vel = []
    for file in types:
        vel.append(np.load(f"{file}_vel.npy"))
    return (torch.from_numpy(v) for v in vel)


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


# class Optim_velocity(nn.Module):
#     def __init__(self, num_years, H, W):
#         super(Optim_velocity, self).__init__()
#         self.v_x = torch.nn.Parameter(torch.randn(num_years, 1, 5, H, W))
#         self.v_y = torch.nn.Parameter(torch.randn(num_years, 1, 5, H, W))

#     def forward(self, data):
#         u_y = torch.gradient(data, dim=3)[0]  # (H,W) --> (y,x)
#         u_x = torch.gradient(data, dim=4)[0]
#         adv = (
#             self.v_x * u_x
#             + self.v_y * u_y
#             + data
#             * (torch.gradient(self.v_y, dim=3)[0] + torch.gradient(self.v_x, dim=4)[0])
#         )
#         out = adv
#         return out, self.v_x, self.v_y


# class Climate_ResNet_2D(nn.Module):

#     def __init__(self, num_channels, layers, hidden_size):
#         super().__init__()
#         layers_cnn = []
#         activation_fns = []
#         self.block = ResidualBlock
#         self.inplanes = num_channels

#         for idx in range(len(layers)):
#             if idx == 0:
#                 layers_cnn.append(
#                     self.make_layer(
#                         self.block, num_channels, hidden_size[idx], layers[idx]
#                     )
#                 )
#             else:
#                 layers_cnn.append(
#                     self.make_layer(
#                         self.block, hidden_size[idx - 1], hidden_size[idx], layers[idx]
#                     )
#                 )

#         self.layer_cnn = nn.ModuleList(layers_cnn)
#         self.activation_cnn = nn.ModuleList(activation_fns)

#     def make_layer(self, block, in_channels, out_channels, reps):
#         layers = []
#         layers.append(block(in_channels, out_channels))
#         self.inplanes = out_channels
#         for i in range(1, reps):
#             layers.append(block(out_channels, out_channels))

#         return nn.Sequential(*layers)

#     def forward(self, data):
#         dx_final = data.float()
#         for l, layer in enumerate(self.layer_cnn):
#             dx_final = layer(dx_final)

#         return dx_final


def nll(mean, std, truth, lat, var_coeff):
    normal_lkl = torch.distributions.normal.Normal(mean, 1e-3 + std)
    lkl = -normal_lkl.log_prob(truth)
    loss_val = lkl.mean() + var_coeff * (std**2).sum()
    return loss_val


def main(
    solver="euler",
    atol=5e-3,
    rtol=5e-3,
    step_size=None,
    niters=100,
    scale=0,
    batch_size=6,
    spectral=0,
    lr=0.0005,
    weight_decay=1e-5,
    dryrun=False,
):
    """
    Main function for ClimODE.

    Args:
        solver (str): The ODE solver to use. Options: ['dopri8', 'dopri5', 'bdf', 'rk4',
            'midpoint', 'adams', 'explicit_adams', 'fixed_adams', 'adaptive_heun', 'euler'].
        atol (float): Absolute tolerance for solver.
        rtol (float): Relative tolerance for solver.
        step_size (float): Optional fixed step size.
        niters (int): Number of iterations.
        scale (int): Scaling parameter (default: 0).
        batch_size (int): Batch size (default: 6).
        spectral (int): Whether to use spectral methods (0 or 1).
        lr (float): Learning rate.
        weight_decay (float): Weight decay factor.
        dryrun (bool): Only run few train/val updates and then exit.
    """
    torch.manual_seed(42)

    SOLVERS = [
        "dopri8",
        "dopri5",
        "bdf",
        "rk4",
        "midpoint",
        "adams",
        "explicit_adams",
        "fixed_adams",
        "adaptive_heun",
        "euler",
    ]

    if solver not in SOLVERS:
        raise ValueError(f"Invalid solver: {solver}. Choose from {SOLVERS}.")

    print("=" * 50)
    print(
        f"Running with solver={solver}, atol={atol}, rtol={rtol},"
        f"step_size={step_size}, niters={niters}, scale={scale},"
        f"batch_size={batch_size}, spectral={spectral}, lr={lr},"
        f"weight_decay={weight_decay}",
    )
    print("=" * 50)
    if dryrun:
        print("=" * 25, "DRYRUN ACTIVE WILL BREAK EARLY", "=" * 25)

    assert torch.cuda.is_available()
    device = torch.device("cuda")

    train_time_scale = slice("2006", "2016")
    val_time_scale = slice("2016", "2016")
    test_time_scale = slice("2017", "2018")
    paths_to_data = [
        "era5_data/geopotential_500/*.nc",
        "era5_data/temperature_850/*.nc",
        "era5_data/2m_temperature/*.nc",
        "era5_data/10m_u_component_of_wind/*.nc",
        "era5_data/10m_v_component_of_wind/*.nc",
    ]
    const_info_path = ["era5_data/constants/constants/constants_5.625deg.nc"]
    levels = ["z", "t", "t2m", "u10", "v10"]

    assert len(paths_to_data) == len(
        levels
    ), "Paths to different type of data must be same as number of types of observations"

    Final_train_data = 0
    Final_val_data = 0
    Final_test_data = 0
    max_lev = []
    min_lev = []

    for idx, data in enumerate(tqdm(paths_to_data, desc="reading data")):
        Train_data, Val_data, Test_data, time_steps, lat, lon, mean, std, time_stamp = (
            get_train_test_data_without_scales_batched(
                data,
                train_time_scale,
                val_time_scale,
                test_time_scale,
                levels[idx],
                spectral,
            )
        )
        max_lev.append(mean)
        min_lev.append(std)
        if idx == 0:
            Final_train_data = Train_data
            Final_val_data = Val_data
            Final_test_data = Test_data
        else:
            Final_train_data = torch.cat([Final_train_data, Train_data], dim=2)
            Final_val_data = torch.cat([Final_val_data, Val_data], dim=2)
            Final_test_data = torch.cat([Final_test_data, Test_data], dim=2)

    print("train, val, test data shapes:")
    print(
        Final_train_data.shape,
        Final_val_data.shape,
        Final_test_data.shape,
    )

    const_channels_info, lat_map, lon_map = add_constant_info(const_info_path)
    H, W = Train_data.shape[3], Train_data.shape[4]
    Train_loader = DataLoader(
        Final_train_data[2:],
        batch_size=batch_size,
    )
    Val_loader = DataLoader(
        Final_val_data[2:],
        batch_size=batch_size,
    )
    Test_loader = DataLoader(Final_test_data[2:], batch_size=batch_size)
    time_loader = DataLoader(time_steps[2:], batch_size=batch_size)
    time_idx_steps = torch.tensor([i for i in range(365 * 4)]).view(-1, 1)
    time_idx = DataLoader(time_idx_steps[2:], batch_size=batch_size)
    num_years = len(range(2006, 2016))

    if not Path("kernel.npy").exists():
        get_gauss_kernel((32, 64), lat, lon)

    kernel = torch.from_numpy(np.load("kernel.npy"))
    if not Path("test_10year_2day_mm_vel.npy").exists():
        print("Fitting velocity...")
        fit_velocity(
            time_idx,
            time_loader,
            Final_train_data,
            Train_loader,
            device,
            num_years,
            paths_to_data,
            scale,
            H,
            W,
            types="train_10year_2day_mm",
            vel_model=Optim_velocity,
            kernel=kernel,
        )
        fit_velocity(
            time_idx,
            time_loader,
            Final_val_data,
            Val_loader,
            device,
            1,
            paths_to_data,
            scale,
            H,
            W,
            types="val_10year_2day_mm",
            vel_model=Optim_velocity,
            kernel=kernel,
        )
        fit_velocity(
            time_idx,
            time_loader,
            Final_test_data,
            Test_loader,
            torch.device("cuda"),
            2,
            paths_to_data,
            scale,
            H,
            W,
            types="test_10year_2day_mm",
            vel_model=Optim_velocity,
            kernel=kernel,
        )

    # vel_train, vel_val = load_velocity(["train_10year_2day_mm", "val_10year_2day_mm"])
    vel_train, vel_val, vel_test = load_velocity(
        ["train_10year_2day_mm", "val_10year_2day_mm", "test_10year_2day_mm"]
    )
    model = Climate_encoder_free_uncertain(
        len(paths_to_data),
        2,
        out_types=len(paths_to_data),
        method=solver,
        use_att=True,
        use_err=True,
        use_pos=False,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, niters)

    const_channels_info = const_channels_info.to(device)
    lat_map = lat_map.to(device)
    lon_map = lon_map.to(device)

    for epoch in range(niters):
        print(f"##### Epoch {epoch} of {niters} #####")
        total_train_loss = 0
        val_loss = 0
        test_loss = 0

        if epoch == 0:
            var_coeff = 0.001
        else:
            var_coeff = 2 * scheduler.get_last_lr()[0]

        pbar = tqdm(
            enumerate(zip(time_loader, Train_loader)),
            total=min(len(time_loader), len(Train_loader)),
            colour="green",
            desc="train",
        )
        vel_train = vel_train.to(device)

        for entry, (time_steps, batch) in pbar:
            if dryrun and entry >= 5:
                break

            batch = batch.to(device)

            optimizer.zero_grad()
            data = batch[0].view(num_years, 1, len(paths_to_data) * (scale + 1), H, W)
            past_sample = vel_train[entry].view(
                num_years, 2 * len(paths_to_data) * (scale + 1), H, W
            )
            model.update_param(
                past_sample,
                const_channels_info,
                lat_map,
                lon_map,
            )
            t = time_steps.float().to(device).flatten()
            mean, std, _ = model(t, data)
            loss = nll(mean, std, batch, lat, var_coeff)
            l2_lambda = 0.001
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            loss = loss + l2_lambda * l2_norm
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": loss.item()})
            assert not torch.isnan(loss), "Quitting due to Nan loss"
            total_train_loss = total_train_loss + loss.item()

        scheduler.step()
        print("|Iter ", epoch, " | Total Train Loss ", total_train_loss, "|")
        optimizer.zero_grad(set_to_none=True)  # Clear memory
        torch.cuda.empty_cache()

        with torch.no_grad():
            vel_val = vel_val.to(device)
            pbar = tqdm(
                enumerate(zip(time_loader, Val_loader)),
                total=min(len(time_loader), len(Val_loader)),
                colour="blue",
                desc="test",
            )
            for entry, (time_steps, batch) in pbar:
                if dryrun and entry >= 10:
                    break
                batch = batch.to(device)
                time_steps = time_steps.to(device).float()

                data = batch[0].view(1, 1, len(paths_to_data) * (scale + 1), H, W)
                past_sample = vel_val[entry].view(
                    1, 2 * len(paths_to_data) * (scale + 1), H, W
                )
                model.update_param(
                    past_sample,
                    const_channels_info,
                    lat_map,
                    lon_map,
                )
                mean, std, _ = model(time_steps, data)
                loss = nll(mean, std, batch, lat, var_coeff)
                assert not torch.isnan(loss), "Quitting due to Nan loss"
                pbar.set_postfix({"val_lss": loss.item()})
                val_loss = val_loss + loss.item()

            print("|Iter ", epoch, " | Total Val Loss ", val_loss, "|")

            print(
                f"Writing model to checkpoints/ClimODE_global_{solver}_{spectral}_model_{epoch}_{val_loss}.pt"
            )
            torch.save(
                model.state_dict(),
                f"checkpoints/ClimODE_global_{solver}_{spectral}_model_{epoch}_{val_loss}.pt",
            )

        if dryrun:
            print("Early break due to dryrun")
            break


if __name__ == "__main__":
    fire.Fire(main)
