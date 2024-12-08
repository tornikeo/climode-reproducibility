
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

from climode.utils import fit_velocity, get_gauss_kernel
from climode.model_function import Optim_velocity, Climate_encoder_free_uncertain
from climode.utils import evaluation_crps_mm, evaluation_acc_mm, evaluation_rmsd_mm


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


def nll(mean, std, truth, lat, var_coeff):
    normal_lkl = torch.distributions.normal.Normal(mean, 1e-3 + std)
    lkl = -normal_lkl.log_prob(truth)
    loss_val = lkl.mean() + var_coeff * (std**2).sum()
    return loss_val


def main(
    checkpoint_path:str,
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
    total_time_len = len(time_steps[2:])
    total_time_steps = time_steps[2:].numpy().flatten().tolist()
    num_years = 2


    # vel_train, vel_val = load_velocity(["train_10year_2day_mm", "val_10year_2day_mm"])
    # vel_train, vel_val, vel_test = load_velocity(
    #     ["train_10year_2day_mm", "val_10year_2day_mm", "test_10year_2day_mm"]
    # )
    vel_test = torch.from_numpy(np.load('test_10year_2day_mm_vel.npy'))
    clim = torch.mean(Final_test_data, dim=0)
    model = Climate_encoder_free_uncertain(
        len(paths_to_data),
        2,
        out_types=len(paths_to_data),
        method=solver,
        use_att=True,
        use_err=True,
        use_pos=False,
    )
    model.load_state_dict(torch.load(checkpoint_path, 
                                    map_location=device, 
                                    weights_only=True))
    model.eval();
    model = model.to(device)
    Lead_RMSD_arr = {
        "z": [[] for _ in range(7)],
        "t": [[] for _ in range(7)],
        "t2m": [[] for _ in range(7)],
        "u10": [[] for _ in range(7)],
        "v10": [[] for _ in range(7)],
    }
    Lead_ACC = {
        "z": [[] for _ in range(7)],
        "t": [[] for _ in range(7)],
        "t2m": [[] for _ in range(7)],
        "u10": [[] for _ in range(7)],
        "v10": [[] for _ in range(7)],
    }
    Lead_CRPS = {
        "z": [[] for _ in range(7)],
        "t": [[] for _ in range(7)],
        "t2m": [[] for _ in range(7)],
        "u10": [[] for _ in range(7)],
        "v10": [[] for _ in range(7)],
    }

    with torch.no_grad():
        vel_test = vel_test.to(device)
        pbar = tqdm(
            enumerate(zip(time_loader, Test_loader)),
            total=len(time_loader),
            colour="blue",
            desc="test",
        )
        const_channels_info = const_channels_info.to(device)
        lat_map = lat_map.to(device)
        lon_map = lon_map.to(device)
        for entry, (time_steps, batch) in pbar:
            if dryrun and entry >= 10:
                break
            batch = batch.to(device)
            time_steps = time_steps.to(device).float()

            data = batch[0].view(num_years, 1, len(paths_to_data) * (scale + 1), H, W)
            past_sample = vel_test[entry].view(
                num_years, 2 * len(paths_to_data) * (scale + 1), H, W
            )
            model.update_param(
                past_sample,
                const_channels_info,
                lat_map,
                lon_map,
            )
            
            mean_pred, std_pred, mean_wo_bias = model(time_steps, data)
            mean_avg = mean_pred.view(-1, len(paths_to_data) * (scale + 1), H, W)
            std_avg = std_pred.view(-1, len(paths_to_data) * (scale + 1), H, W)
            for yr in range(2):
                for t_step in range(1, len(time_steps), 1):
                    evaluate_rmsd = evaluation_rmsd_mm(
                        mean_pred[t_step, yr, :, :, :].cpu(),
                        batch[t_step, yr, :, :, :].cpu(),
                        lat,
                        lon,
                        max_lev,
                        min_lev,
                        H,
                        W,
                        levels,
                    )
                    evaluate_acc = evaluation_acc_mm(
                        mean_pred[t_step, yr, :, :, :].cpu(),
                        batch[t_step, yr, :, :, :].cpu(),
                        lat,
                        lon,
                        max_lev,
                        min_lev,
                        H,
                        W,
                        levels,
                        clim[yr, :, :, :].cpu().detach().numpy(),
                    )
                    evaluate_crps = evaluation_crps_mm(
                        mean_pred[t_step, yr, :, :, :].cpu(),
                        batch[t_step, yr, :, :, :].cpu(),
                        lat,
                        lon,
                        max_lev,
                        min_lev,
                        H,
                        W,
                        levels,
                        std_pred[t_step, yr, :, :, :].cpu(),
                    )
                    for idx, lev in enumerate(levels):
                        Lead_RMSD_arr[lev][t_step - 1].append(evaluate_rmsd[idx])
                        Lead_ACC[lev][t_step - 1].append(evaluate_acc[idx])
                        Lead_CRPS[lev][t_step - 1].append(evaluate_crps[idx])
    for t_idx in range(6):
        for idx, lev in enumerate(levels):
            print("Lead Time ",(t_idx+1)*6, "hours ","| Observable ",lev, "| Mean RMSD ", np.mean(Lead_RMSD_arr[lev][t_idx]), "| Std RMSD ", np.std(Lead_RMSD_arr[lev][t_idx]))
            print("Lead Time ",(t_idx+1)*6, "hours ","| Observable ",lev, "| Mean ACC ", np.mean(Lead_ACC[lev][t_idx]), "| Std ACC ", np.std(Lead_ACC[lev][t_idx]))
            print("Lead Time ",(t_idx+1)*6, "hours ","| Observable ",lev, "| Mean CRPS ", np.mean(Lead_CRPS[lev][t_idx]), "| Std CRPS ", np.std(Lead_CRPS[lev][t_idx]))

if __name__ == "__main__":
    fire.Fire(main)