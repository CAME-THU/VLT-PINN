"""Data fitting by NN. The PDE losses are absent. Only u, v are considered."""
import deepxde as dde
import numpy as np
# import torch
import os
import argparse

# # relative import is not suggested
# # set projects_deepxde as root (by clicking), or use sys.path.insert like this (then ignore the red errors):
# import sys
# sys.path.insert(0, os.path.dirname("E:/Research_ASUS/1 PhD project/AI_PDE/projects_PINN/"))
from configs.maps_uv import Maps
from configs.post_shear import PostProcessShear
from utils.utils import efmt, cal_stat
from utils.dataset_modi import ScaledDataSet


def main(args):
    case_name = args.case_name
    if case_name == "jet_lam_plane":
        from casedef.case_jet_lam_pln import Case
    elif case_name == "jet_lam_round":
        from casedef.case_jet_lam_rnd import Case
    elif case_name == "jet_tur_plane":
        from casedef.case_jet_tur_pln import Case
    elif case_name == "jet_tur_round":
        from casedef.case_jet_tur_rnd import Case
    elif case_name == "wake_lam_plane":
        from casedef.case_wake_lam_pln import Case
    elif case_name == "wake_lam_round":
        from casedef.case_wake_lam_rnd import Case
    elif case_name == "mixing_layer":
        from casedef.case_mix import Case
    elif case_name == "boundary_layer":
        from casedef.case_bl import Case
    else:
        Case = None
    case = Case(args)

    scale_u, scale_v = args.scales["u"], args.scales["v"]
    scale_x, scale_y = args.scales["x"], args.scales["y"]
    shift_x, shift_y = args.shifts["x"], args.shifts["y"]

    x_l, x_r, y_l, y_r = case.x_l, case.x_r, case.y_l, case.y_r

    # ----------------------------------------------------------------------
    # define observation points
    spc_ob = 0.02  # m
    # spc_ob = (y_r - y_l) / 20  # m
    n_ob_x, n_ob_y = int((x_r - x_l) / spc_ob) + 1, int((y_r - y_l) / spc_ob) + 1
    ob_xx, ob_yy = np.meshgrid(np.linspace(x_l, x_r, n_ob_x),
                                   np.linspace(y_l, y_r, n_ob_y), indexing="ij")
    ob_xy = np.vstack((np.ravel(ob_xx), np.ravel(ob_yy))).T
    n_ob = n_ob_x * n_ob_y

    ob_u = case.func_u(ob_xy)
    ob_v = case.func_v(ob_xy)
    normal_noise_u = np.random.randn(len(ob_u))[:, None]
    normal_noise_v = np.random.randn(len(ob_v))[:, None]
    ob_u += normal_noise_u * ob_u * args.noise_level
    ob_v += normal_noise_v * ob_v * args.noise_level

    test_xx, test_yy = np.meshgrid(np.linspace(x_l, x_r, n_ob_x * 4),
                                   np.linspace(y_l, y_r, n_ob_y * 4), indexing="ij")
    test_xy = np.vstack((np.ravel(test_xx), np.ravel(test_yy))).T
    test_u = case.func_u(test_xy)
    test_v = case.func_v(test_xy)

    # data = dde.data.DataSet(
    data = ScaledDataSet(
        X_train=ob_xy,
        y_train=np.hstack([ob_u, ob_v]),
        X_test=test_xy,
        y_test=np.hstack([test_u, test_v]),
        scales=(scale_u, scale_v),
        # standardize=True,
    )

    # ----------------------------------------------------------------------
    # define maps (network and input/output transforms)
    maps = Maps(args=args, case=case)
    net = maps.net
    model = dde.Model(data, net)

    # ----------------------------------------------------------------------
    # define training and train
    output_dir = f"../results/{case_name}/uv_fit/"
    output_dir += f"ob{n_ob}-N{efmt(args.noise_level)}"
    # output_dir += "_{:.1e}-{:.1e}_{:.1e}-{:.1e}/".format(scale_u, scale_v, scale_x, scale_y)

    output_dir += f"_u{efmt(scale_u)}-v{efmt(scale_v)}" \
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
    pp2d = PostProcessShear(args=args, case=case, model=model, output_dir=output_dir)
    pp2d.save_data()
    pp2d.save_metrics()
    # pp2d.plot_save_loss_history()
    # if len(args.infer_paras) > 0:
    #     pp2d.save_para_metrics()
    #     pp2d.plot_para_history(var_saver)
    pp2d.delete_old_models()
    figsize1 = (10, 4) if args.case_name == "mixing_layer" else (12, 3)
    figsize2 = (10, 8.2) if args.case_name == "mixing_layer" else (10, 5)
    # pp2d.plot_sampling_points(figsize=figsize1)
    pp2d.plot_2dfields(figsize=figsize1)
    pp2d.plot_2dfields_comp(figsize=figsize2, is_vertical=True)
    pp2d.plot_1dcurves(select_x=(0.02, 0.5, 1.0), select_y=(0.00, 0.02, 0.05))
    # if type(net) == dde.nn.pytorch.fnn.FNN:
    #     pp2d.plot_lossgrads_fnn_uv()

    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyper-parameters")
    parser.add_argument("--case_name", type=str, default="jet_lam_plane")

    # parser.add_argument("--problem_type", type=str, default="forward", help="options: forward, inverse, inverse_nu")
    parser.add_argument("--problem_type", type=str, default="fit")
    parser.add_argument("--bc_type", type=str, default="none", help="options: none, soft, hard_LR, hard_DU")
    parser.add_argument("--oc_type", type=str, default="soft", help="options: none, soft")

    parser.add_argument("--scales", type=dict,
                        default={"x": 1.0, "y": 1.0, "u": 1.0, "v": 1.0},
                        help="(variables * scale) for NN I/O, PDE scaling, and parameter inference")
    parser.add_argument("--shifts", type=dict, default={"x": 0.0, "y": 0.0},
                        help="((independent variables + shift) * scale) for NN input and PDE scaling")

    # parser.add_argument("--infer_paras", type=dict, default={},
    #                     help="initial values for unknown physical parameters to be inferred")
    parser.add_argument("--noise_level", type=float, default=0.00,
                        help="noise level of observed data for inverse problems, such as 0.02 (2%)")

    parser.add_argument("--n_iter", type=int, default=20000, help="number of training iterations")
    parser.add_argument("--i_run", type=int, default=1, help="index of the current run")

    # ----------------------------------------------------------------------
    # set arguments
    args = parser.parse_args()

    case_dict = {
        1: "jet_lam_plane",
        2: "jet_lam_round",
        3: "jet_tur_plane",
        4: "jet_tur_round",
        5: "wake_lam_plane",
        6: "wake_lam_round",
        7: "mixing_layer",
        8: "boundary_layer",
    }
    trans_para_dict = {
        "jet_lam_plane": (100, 1000, 100, 100),
        "jet_lam_round": (100, 1000, 100, 100),
        "jet_tur_plane": (1, 10, 10, 100),
        "jet_tur_round": (1, 100, 10, 100),
        "wake_lam_plane": (100, 10000, 1, 100),
        "wake_lam_round": (100, 10000, 1, 100),
        "mixing_layer": (10, 1000, 1, 10),
        "boundary_layer": (10, 1000, 1, 10),
    }  # scale_u, scale_v, scale_x, scale_y

    args.n_iter = 20000

    # ----------------------------------------------------------------------
    # run
    for case_id in (1, ):
    # for case_id in (1, 2, 5, 6, 7, 8, ):
        args.case_name = case_dict[case_id]
        args.scales["u"], args.scales["v"], args.scales["x"], args.scales["y"] = trans_para_dict[args.case_name]

        n_run = 1

        for args.i_run in range(1, 1 + n_run):
            output_dir = main(args)
        # cal_stat(output_dir[:-2], n_run)
