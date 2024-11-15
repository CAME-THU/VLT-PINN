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
from utils.dataset_modi import ScaledDataSet


def main(args):
    case = Case(args)
    case_name = f"Re{case.Re}_a{case.a}_b{case.b}_m{case.m}_{case.x_l}_{case.x_r}_{case.y_l}_{case.y_r}"

    x_l, x_r, y_l, y_r = case.x_l, case.x_r, case.y_l, case.y_r

    scale_u, scale_v, scale_p = args.scales["u"], args.scales["v"], args.scales["p"]
    scale_x, scale_y = args.scales["x"], args.scales["y"]
    shift_x, shift_y = args.shifts["x"], args.shifts["y"]

    # ----------------------------------------------------------------------
    # define observation points
    spc_ob = 0.02  # m
    # spc_ob = (x_r - x_l) / 20  # m
    n_ob_x, n_ob_y = int((x_r - x_l) / spc_ob) + 1, int((y_r - y_l) / spc_ob) + 1
    ob_xx, ob_yy = np.meshgrid(np.linspace(x_l, x_r, n_ob_x),
                                   np.linspace(y_l, y_r, n_ob_y), indexing="ij")
    ob_xy = np.vstack((np.ravel(ob_xx), np.ravel(ob_yy))).T
    n_ob = n_ob_x * n_ob_y

    ob_u = case.func_u(ob_xy)
    ob_v = case.func_v(ob_xy)
    ob_p = case.func_p(ob_xy)
    normal_noise_u = np.random.randn(len(ob_u))[:, None]
    normal_noise_v = np.random.randn(len(ob_v))[:, None]
    normal_noise_p = np.random.randn(len(ob_p))[:, None]
    ob_u += normal_noise_u * ob_u * args.noise_level
    ob_v += normal_noise_v * ob_v * args.noise_level
    ob_p += normal_noise_p * ob_p * args.noise_level

    test_xx, test_yy = np.meshgrid(np.linspace(x_l, x_r, n_ob_x * 4),
                                   np.linspace(y_l, y_r, n_ob_y * 4), indexing="ij")
    test_xy = np.vstack((np.ravel(test_xx), np.ravel(test_yy))).T
    test_u = case.func_u(test_xy)
    test_v = case.func_v(test_xy)
    test_p = case.func_p(test_xy)

    # data = dde.data.DataSet(
    data = ScaledDataSet(
        X_train=ob_xy,
        y_train=np.hstack([ob_u, ob_v, ob_p]),
        X_test=test_xy,
        y_test=np.hstack([test_u, test_v, test_p]),
        scales=(scale_u, scale_v, scale_p),
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
    # restore the best model (do not if using LBFGS)
    model_list = os.listdir(output_dir + "models/")
    model_list_better = [s for s in model_list if "better" in s]
    saved_epochs = [int(s.split("-")[1][:-3]) for s in model_list_better]
    best_epoch = max(saved_epochs)
    model.restore(output_dir + f"models/model_better-{best_epoch}.pt")

    # ----------------------------------------------------------------------
    # post-process
    pp2d = PostProcessKov(args=args, case=case, model=model, output_dir=output_dir)
    pp2d.save_data()
    pp2d.save_metrics()
    # pp2d.plot_save_loss_history()
    # if args.problem_type == "inverse_nu":
    #     pp2d.save_var_metrics((case.nu, ), (case.nu_infe_s / args.scales["nu"], ), ("nu", ))
    #     pp2d.plot_save_var_history((case.nu, ), (args.scales["nu"], ), (r"$\nu$", ), ("nu", ), ("m$^2$/s", ))
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

    # parser.add_argument("--problem_type", type=str, default="inverse", help="options: forward, inverse")
    parser.add_argument("--problem_type", type=str, default="fit")
    parser.add_argument("--bc_type", type=str, default="none", help="options: none, soft")
    parser.add_argument("--oc_type", type=str, default="soft", help="options: none, soft")

    parser.add_argument("--scales", type=dict,
                        default={"x": 1.0, "y": 1.0, "u": 1.0, "v": 1.0, "p": 1.0},
                        help="(variables * scale) for NN I/O, PDE scaling, and parameter inference")
    parser.add_argument("--shifts", type=dict, default={"x": 0.0, "y": 0.0},
                        help="((independent variables + shift) * scale) for NN input and PDE scaling")

    parser.add_argument("--infer_paras", type=dict, default={},
                        help="initial values for unknown physical parameters to be inferred")
    parser.add_argument("--noise_level", type=float, default=0.00,
                        help="noise level of observed data for inverse problems, such as 0.02 (2%)")

    parser.add_argument("--n_iter", type=int, default=20000, help="number of training iterations")
    parser.add_argument("--i_run", type=int, default=1, help="index of the current run")

    # ----------------------------------------------------------------------
    # set arguments
    args = parser.parse_args()

    trans_para_dict = {
        1: (1, 0.01, 10, 100, 1, 0, 0),
        # 2: (1, 0.1, 1, 10, 1, -0.2, 0),  # a, b, m, nu = 1e-4, 0.25, 0, 0.02
        2: (1, 0.01, 1, 100, 1, -0.2, 0),  # a, b, m, nu = 1e-17, 0.25, 0, 0.005
    }  # scale_u, scale_v, scale_p, scale_x, scale_y, shift_x, shift_y

    args.n_iter = 20000

    # ----------------------------------------------------------------------
    # run
    for args.case_id in (1, ):
    # for args.case_id in (1, 2,):
        args.scales["u"], args.scales["v"], args.scales["p"], args.scales["x"], args.scales["y"], \
        args.shifts["x"], args.shifts["y"] = trans_para_dict[args.case_id]

        n_run = 1
        for args.i_run in range(1, 1 + n_run):
            main(args)

