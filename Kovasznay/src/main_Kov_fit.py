"""Data fitting by NN. The PDE losses are absent."""
import deepxde as dde
import numpy as np
# import torch
import os
import argparse

# # relative import is not suggested
# # set projects_deepxde as root (by clicking), or use sys.path.insert like this (then ignore the red errors):
# import sys
# sys.path.insert(0, os.path.dirname("E:/Research_ASUS/1 PhD project/AI_PDE/projects_PINN/"))
from configs.case_Kov import Case
from configs.maps_uvp import Maps
from configs.post_Kov import PostProcessKov
from utils.utils import efmt, cal_stat


def main(args):
    case = Case(args)
    case_name = f"Re{case.Re}_a{case.a}_b{case.b}_m{case.m}_{case.x_l}_{case.x_r}_{case.y_l}_{case.y_r}"

    # ----------------------------------------------------------------------
    # define constants
    scale_u, scale_v, scale_p = args.scale_u, args.scale_v, args.scale_p
    scale_x, scale_y = args.scale_x, args.scale_y
    shift_x, shift_y = args.shift_x, args.shift_y

    x_l, x_r, y_l, y_r = case.x_l, case.x_r, case.y_l, case.y_r
    xs_l, xs_r = scale_x * (x_l + shift_x), scale_x * (x_r + shift_x)
    ys_l, ys_r = scale_y * (y_l + shift_y), scale_y * (y_r + shift_y)

    # ----------------------------------------------------------------------
    # define observation points
    spc_ob = 0.02  # m
    # spc_ob = (x_r - x_l) / 20  # m
    n_ob_x, n_ob_y = int((x_r - x_l) / spc_ob) + 1, int((y_r - y_l) / spc_ob) + 1
    ob_xsxs, ob_ysys = np.meshgrid(np.linspace(xs_l, xs_r, n_ob_x),
                                   np.linspace(ys_l, ys_r, n_ob_y), indexing="ij")
    ob_xy_s = np.vstack((np.ravel(ob_xsxs), np.ravel(ob_ysys))).T
    n_ob = n_ob_x * n_ob_y

    ob_us = case.func_us(ob_xy_s)
    ob_vs = case.func_vs(ob_xy_s)
    ob_ps = case.func_ps(ob_xy_s)
    normal_noise_u = np.random.randn(len(ob_us))[:, None]
    normal_noise_v = np.random.randn(len(ob_vs))[:, None]
    normal_noise_p = np.random.randn(len(ob_ps))[:, None]
    ob_us += normal_noise_u * ob_us * args.noise_level
    ob_vs += normal_noise_v * ob_vs * args.noise_level
    ob_ps += normal_noise_p * ob_ps * args.noise_level

    ts_xsxs, ts_ysys = np.meshgrid(np.linspace(xs_l, xs_r, n_ob_x * 4),
                                   np.linspace(ys_l, ys_r, n_ob_y * 4), indexing="ij")
    ts_xy_s = np.vstack((np.ravel(ts_xsxs), np.ravel(ts_ysys))).T
    ts_us = case.func_us(ts_xy_s)
    ts_vs = case.func_vs(ts_xy_s)
    ts_ps = case.func_ps(ts_xy_s)

    data = dde.data.DataSet(
        X_train=ob_xy_s,
        y_train=np.hstack([ob_us, ob_vs, ob_ps]),
        X_test=ts_xy_s,
        y_test=np.hstack([ts_us, ts_vs, ts_ps]),
        # standardize=True,
    )

    # ----------------------------------------------------------------------
    # define maps (network and input/output transforms)
    maps = Maps(args=args, case=case)
    net = maps.net
    model = dde.Model(data, net)

    # ----------------------------------------------------------------------
    # define training and train
    output_dir = f"../results/{case_name}/fit/"
    output_dir += f"ob{n_ob}-N{efmt(args.noise_level)}"
    # output_dir += "_{:.1e}-{:.1e}_{:.1e}-{:.1e}/".format(scale_u, scale_v, scale_x, scale_y)

    # output_dir += f"_{efmt(scale_u)}-{efmt(scale_v)}_" \
    #               f"{efmt(scale_x)}-{efmt(scale_y)}_{efmt(shift_x)}-{efmt(shift_y)}"
    output_dir += f"_u{efmt(scale_u)}-v{efmt(scale_v)}-p{efmt(scale_p)}" \
                  f"_x{efmt(scale_x)}-y{efmt(scale_y)}_x{efmt(shift_x)}-y{efmt(shift_y)}"
    i_run = args.i_run
    while True:
        if not os.path.exists(output_dir + f"/{i_run}/"):
            output_dir += f"/{i_run}/"
            os.makedirs(output_dir)
            os.makedirs(output_dir + "models/")
            break
        else:
            i_run += 1

    model_saver = dde.callbacks.ModelCheckpoint(
        output_dir + "models/model_better", save_better_only=True, period=100)
    callbacks = [model_saver, ]
    # resampler = dde.callbacks.PDEPointResampler(period=100)
    # callbacks += [resampler, ]

    loss_weights = None
    # loss_weights = [100, 100, 1, 1, 1]
    model.compile(optimizer="adam",  # "sgd", "rmsprop", "adam", "adamw"
                  lr=1e-3,
                  loss="MSE",
                  decay=("step", 1000, 0.8),
                  loss_weights=loss_weights,
                  )
    model.train(iterations=args.n_iter,
                display_every=100,
                disregard_previous_best=False,
                callbacks=callbacks,
                model_restore_path=None,
                model_save_path=output_dir + "models/model_last",)

    # ----------------------------------------------------------------------
    # restore a new model
    # model.restore(output_dir + "models/model_last-20000.pt")

    # ----------------------------------------------------------------------
    # post-process
    pp2d = PostProcessKov(args=args, case=case, model=model, output_dir=output_dir)
    pp2d.save_data()
    pp2d.save_metrics()
    # pp2d.plot_save_loss_history()
    # if problem_type == "inverse_nu":
    #     pp2d.save_var_metrics((case.nu, ), (case.var_nu_s / args.scale_nu, ), ("nu", ))
    #     pp2d.plot_save_var_history((case.nu, ), (args.scale_nu, ), (r"$\nu$", ), ("nu", ), ("m$^2$/s", ))
    pp2d.delete_old_models()
    # pp2d.plot_sampling_points(figsize=(4, 8))
    pp2d.plot_2dfields(figsize=(4, 8))
    pp2d.plot_2dfields_comp(figsize=(5, 8))
    pp2d.plot_1dcurves(select_x=(x_l + 0.0 * (x_r - x_l), x_l + 0.95 * (x_r - x_l), x_l + 1.0 * (x_r - x_l)),
                       select_y=(y_l + 0.0 * (y_r - y_l), y_l + 0.50 * (y_r - y_l), y_l + 1.0 * (y_r - y_l)))
    # if type(net) == dde.nn.pytorch.fnn.FNN:
    #     pp2d.plot_lossgrads_fnn_uv()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyper-parameters")
    parser.add_argument("--case_id", type=int, default=1)
    # parser.add_argument("--problem_type", type=str, default="forward", help="options: forward, inverse, inverse_nu")
    parser.add_argument("--problem_type", type=str, default="fit")

    parser.add_argument("--scale_u", type=float, default=1.0)
    parser.add_argument("--scale_v", type=float, default=1.0)
    parser.add_argument("--scale_p", type=float, default=1.0)
    parser.add_argument("--scale_x", type=float, default=1.0)
    parser.add_argument("--scale_y", type=float, default=1.0)
    parser.add_argument("--shift_x", type=float, default=0.0)
    parser.add_argument("--shift_y", type=float, default=0.0)

    parser.add_argument("--noise_level", type=float, default=0.02,
                        help="noise level for observations, such as 0.02 (2%)")

    parser.add_argument("--n_iter", type=int, default=20000)
    parser.add_argument("--i_run", type=int, default=1)

    # ----------------------------------------------------------------------
    # set arguments
    args = parser.parse_args()

    para_dict = {
        1: (1, 0.01, 10, 100, 1, 0, 0),
        # 2: (1, 0.1, 1, 10, 1, -0.2, 0),  # a, b, m, nu = 1e-4, 0.25, 0, 0.02
        2: (1, 0.01, 1, 100, 1, -0.2, 0),  # a, b, m, nu = 1e-17, 0.25, 0, 0.005
    }  # scale_u, scale_v, scale_p, scale_x, scale_y, shift_x, shift_y

    args.n_iter = 20000

    # ----------------------------------------------------------------------
    # run
    for args.case_id in (1, 2,):
        args.scale_u, args.scale_v, args.scale_p, args.scale_x, args.scale_y, \
        args.shift_x, args.shift_y = para_dict[args.case_id]

        n_run = 3

        for args.i_run in range(1, 1 + n_run):
            main(args)

