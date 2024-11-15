"""PINN to solve shear flows. Only u, v are solved."""
import deepxde as dde
import numpy as np
import torch
import os
import argparse

# # relative import is not suggested
# # set projects_deepxde as root (by clicking), or use sys.path.insert like this (then ignore the red errors):
# import sys
# sys.path.insert(0, os.path.dirname("E:/Research_ASUS/1 PhD project/AI_PDE/projects_PINN/"))
from configs.points_shear import MyPoints
from configs.maps_uv import Maps
from configs.post_shear import PostProcessShear
from utils.utils import efmt, cal_stat
from utils.callbacks_modi import VariableSaver


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

    # ----------------------------------------------------------------------
    # define sampling points
    n_dmn = 2000
    n_bdr = 400
    data = dde.data.PDE(
        case.geom,
        case.pde_uv,
        case.icbcocs,
        num_domain=n_dmn,
        num_boundary=n_bdr,
        train_distribution="pseudo",  # "Hammersley", "uniform", "pseudo"
        # anchors=ob_xy_s,
        # solution=func_sol,
        num_test=1000,
    )
    my_points = MyPoints(args=args, case=case, ob_xy=case.ob_xy, dtype=data.train_x.dtype)
    data = my_points.execute_modify(data=data)
    n_dmn, n_bdr = my_points.n_dmn, my_points.n_bdr
    n_bdr = 0 if args.bc_type == "none" else n_bdr

    # ----------------------------------------------------------------------
    # define maps (network and input/output transforms)
    maps = Maps(args=args, case=case)
    net = maps.net
    model = dde.Model(data, net)

    # ----------------------------------------------------------------------
    # define training and train
    output_dir = f"../results/{case_name}/uv_{args.problem_type}/"
    output_dir += f"dmn{n_dmn}"
    if args.bc_type == "soft":
        output_dir += f"_bdr{n_bdr}"
    if args.oc_type == "soft":
        output_dir += f"_ob{case.n_ob}-N{efmt(args.noise_level)}"
    # output_dir += "_{:.1e}-{:.1e}_{:.1e}-{:.1e}/".format(scale_u, scale_v, scale_x, scale_y)

    output_dir += f"_u{efmt(scale_u)}-v{efmt(scale_v)}" \
                  f"_x{efmt(scale_x)}-y{efmt(scale_y)}_x{efmt(shift_x)}-y{efmt(shift_y)}"
    if "nu" in args.infer_paras:
        output_dir += f"_nu{efmt(args.scales['nu'])}"

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

    external_trainable_variables, para_dict, var_saver = [], {}, None
    if "nu" in args.infer_paras:
        external_trainable_variables += [case.nu_infe_s, ]
        para_dict["nu"] = case.nu_infe_s
    if len(para_dict) > 0:
        var_saver = VariableSaver(para_dict, args.scales, period=100, filename=output_dir + "parameters_history.csv")
        callbacks.append(var_saver)

    loss_weights = None
    # loss_weights = [100, 100, 1, 1, 1]
    model.compile(optimizer="adam",  # "sgd", "rmsprop", "adam", "adamw"
                  lr=1e-3,
                  loss="MSE",
                  decay=("step", 1000, 0.8),
                  loss_weights=loss_weights,
                  external_trainable_variables=external_trainable_variables,
                  )
    print("[" + ", ".join(case.names["equations"] + case.names["ICBCOCs"]) + "]" + "\n")
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
    pp2d.plot_save_loss_history()
    if len(args.infer_paras) > 0:
        pp2d.save_para_metrics()
        pp2d.plot_para_history(var_saver)
    pp2d.delete_old_models()
    figsize1 = (10, 4) if args.case_name == "mixing_layer" else (12, 3)
    figsize2 = (10, 8.2) if args.case_name == "mixing_layer" else (10, 5)
    pp2d.plot_sampling_points(figsize=figsize1)
    pp2d.plot_2dfields(figsize=figsize1)
    pp2d.plot_2dfields_comp(figsize=figsize2, is_vertical=True)
    pp2d.plot_1dcurves(select_x=(0.02, 0.5, 1.0), select_y=(0.00, 0.02, 0.05))
    # if type(net) == dde.nn.pytorch.fnn.FNN:
    #     pp2d.plot_lossgrads_fnn_uv()

    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyper-parameters")
    parser.add_argument("--case_name", type=str, default="jet_lam_plane")

    parser.add_argument("--problem_type", type=str, default="forward", help="options: forward, inverse")
    parser.add_argument("--bc_type", type=str, default="soft", help="options: none, soft, hard_LR, hard_DU")
    parser.add_argument("--oc_type", type=str, default="none", help="options: none, soft")

    parser.add_argument("--scales", type=dict,
                        default={"x": 1.0, "y": 1.0, "u": 1.0, "v": 1.0},
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

    args.problem_type, args.bc_type, args.oc_type = "forward", "soft", "none"

    # args.problem_type, args.bc_type, args.oc_type = "inverse", "none", "soft"
    # args.infer_paras["nu"], args.scales["nu"] = 1e-5, 1e4

    # args.n_iter = 200
    # args.n_iter = 20000
    args.n_iter = 30000

    # ----------------------------------------------------------------------
    # run
    for case_id in (6, ):
        args.case_name = case_dict[case_id]
        args.scales["u"], args.scales["v"], args.scales["x"], args.scales["y"] = trans_para_dict[args.case_name]

        n_run = 1
        for args.i_run in range(1, 1 + n_run):
            output_dir = main(args)
        # cal_stat(output_dir[:-2], n_run)

        # for args.scales["x"] in (10, 100):
        #     for args.i_run in range(1, 1 + n_run):
        #         main(args)

        # # mixing layer
        # for args.scales["u"] in (1/0.15, 10, 100, 1000):
        #     for args.scales["v"] in (1/0.15, 100, 1000, 10000):
        #         for args.i_run in range(1, 1 + n_run):
        #             main(args)
        # for args.shifts["x"] in (0, -0.5):
        #     for args.shifts["y"] in (0, 0.2):
        #         if args.shifts["x"] == 0 and args.shifts["y"] == 0:
        #             continue
        #         else:
        #             for args.i_run in range(1, 1 + n_run):
        #                 main(args)


