"""PINN to solve shear flows, with temperature (T) equation being solved (only case 1-4 have T equation)."""
import deepxde as dde
import numpy as np
# import torch
import os
import argparse

# # relative import is not suggested
# # set projects_deepxde as root (by clicking), or use sys.path.insert like this (then ignore the red errors):
# import sys
# sys.path.insert(0, os.path.dirname("E:/Research_ASUS/1 PhD project/AI_PDE/projects_PINN/"))
from configs.cal_domain import RectCalDomain
from configs.points_shear import MyPoints
from configs.maps_uvT import Maps
from configs.post_shear import PostProcessShear
from utils.utils import efmt, cal_stat


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
    else:
        Case = None
    case = Case(args)

    # ----------------------------------------------------------------------
    # define constants
    problem_type = args.problem_type  # "forward", "inverse", "inverse_nu"
    if problem_type == "forward":
        bc_type = "soft"  # "none", "soft", "hard_LR", "hard_DU"
        oc_type = "none"  # "none", "soft"
    else:
        bc_type = "none"  # "none", "soft", "hard_LR", "hard_DU"
        oc_type = "soft"  # "none", "soft"

    scale_u, scale_v, scale_T = args.scale_u, args.scale_v, args.scale_T
    scale_x, scale_y = args.scale_x, args.scale_y
    shift_x, shift_y = args.shift_x, args.shift_y

    x_l, x_r, y_l, y_r = case.x_l, case.x_r, case.y_l, case.y_r
    xs_l, xs_r = scale_x * (x_l + shift_x), scale_x * (x_r + shift_x)
    ys_l, ys_r = scale_y * (y_l + shift_y), scale_y * (y_r + shift_y)

    # ----------------------------------------------------------------------
    # define pde
    pde = case.pde_uvT
    case.names["dependents"] += ["T"]
    case.names["equations"] += ["energy"]

    # ----------------------------------------------------------------------
    # define calculation domain (geometry and boundaries), ICs, BCs, OCs
    domain = RectCalDomain(xymin=[xs_l, ys_l], xymax=[xs_r, ys_r])
    geom = domain.geom

    # bc_left_u = dde.icbc.DirichletBC(geom, case.func_us, domain.bdr_down, component=0)
    # bc_left_v = dde.icbc.DirichletBC(geom, case.func_vs, domain.bdr_down, component=1)
    # bc_far_u = dde.icbc.DirichletBC(geom, case.func_us, domain.bdr_up, component=0)
    # bc_far_v = dde.icbc.DirichletBC(geom, case.func_vs, domain.bdr_up, component=1)

    # bc_sym_dudy = dde.icbc.NeumannBC(geom, lambda xy_s: 0, domain.bdr_down, component=1)  # unknown bug
    bc_sym_dudy = dde.icbc.OperatorBC(
        geom, lambda xy_s, uvT_s, _: dde.grad.jacobian(uvT_s, xy_s, i=0, j=1), domain.bdr_down)
    bc_sym_v = dde.icbc.DirichletBC(geom, lambda xy_s: 0, domain.bdr_down, component=1)

    bc_all_u = dde.icbc.DirichletBC(geom, case.func_us, lambda _, on_boundary: on_boundary, component=0)
    bc_all_v = dde.icbc.DirichletBC(geom, case.func_vs, lambda _, on_boundary: on_boundary, component=1)
    bc_left_right_u = dde.icbc.DirichletBC(geom, case.func_us, domain.bdr_left_right, component=0)
    bc_left_right_v = dde.icbc.DirichletBC(geom, case.func_vs, domain.bdr_left_right, component=1)
    bc_down_up_u = dde.icbc.DirichletBC(geom, case.func_us, domain.bdr_down_up, component=0)
    bc_down_up_v = dde.icbc.DirichletBC(geom, case.func_vs, domain.bdr_down_up, component=1)
    bc_left_right_up_u = dde.icbc.DirichletBC(geom, case.func_us, domain.bdr_left_right_up, component=0)
    bc_left_right_up_v = dde.icbc.DirichletBC(geom, case.func_vs, domain.bdr_left_right_up, component=1)

    # bc_left_T = dde.icbc.DirichletBC(geom, case.func_Ts, domain.bdr_left, component=2)
    bc_left_right_up_T = dde.icbc.DirichletBC(geom, case.func_Ts, domain.bdr_left_right_up, component=2)
    bc_sym_dTdy = dde.icbc.OperatorBC(
        geom, lambda xy_s, uvT_s, _: dde.grad.jacobian(uvT_s, xy_s, i=2, j=1), domain.bdr_down)

    if bc_type == "soft":
        case.icbcocs += [bc_left_right_up_u, bc_sym_dudy, bc_all_v, bc_left_right_up_T, bc_sym_dTdy]
        case.names["ICBCOCs"] = ["BC_left_right_up_u", "BC_sym_dudy", "BC_all_v", "BC_left_right_up_T", "BC_sym_dTdy"]
        # if problem_type == "forward":
        #     case.icbcocs += [bc_left_right_up_u, bc_sym_dudy, bc_all_v, bc_left_right_up_T, bc_sym_dTdy]
        #     case.names["ICBCOCs"] = ["BC_left_right_up_u", "BC_sym_dudy", "BC_all_v", "BC_left_right_up_T", "BC_sym_dTdy"]
        # else:
        #     case.icbcocs += [bc_sym_dudy, bc_sym_v]
        #     case.names["ICBCOCs"] = ["BC_sym_dudy", "BC_sym_v"]
    elif bc_type == "hard_LR":
        case.icbcocs += [bc_down_up_u, bc_down_up_v]
        case.names["ICBCOCs"] = ["BC_down_up_u", "BC_down_up_v"]
    elif bc_type == "hard_DU":
        case.icbcocs += [bc_left_right_u, bc_left_right_v]
        case.names["ICBCOCs"] = ["BC_left_right_u", "BC_left_right_v"]
    else:  # "none"
        pass

    if oc_type == "soft":
        spc_ob = 0.01  # m
        # spc_ob = (y_r - y_l) / 20  # m
        n_ob_x, n_ob_y = int((x_r - x_l) / spc_ob) + 1, int((y_r - y_l) / spc_ob) + 1
        ob_xsxs, ob_ysys = np.meshgrid(np.linspace(xs_l, xs_r, n_ob_x),
                                       np.linspace(ys_l, ys_r, n_ob_y), indexing="ij")
        ob_xy_s = np.vstack((np.ravel(ob_xsxs), np.ravel(ob_ysys))).T
        n_ob = n_ob_x * n_ob_y

        ob_us = case.func_us(ob_xy_s,)
        ob_vs = case.func_vs(ob_xy_s)
        ob_Ts = case.func_Ts(ob_xy_s)
        normal_noise_u = np.random.randn(len(ob_us))[:, None]
        normal_noise_v = np.random.randn(len(ob_vs))[:, None]
        normal_noise_T = np.random.randn(len(ob_Ts))[:, None]
        ob_us += normal_noise_u * ob_us * args.noise_level
        ob_vs += normal_noise_v * ob_vs * args.noise_level
        ob_Ts += normal_noise_T * ob_Ts * args.noise_level

        oc_u = dde.icbc.PointSetBC(ob_xy_s, ob_us, component=0)
        oc_v = dde.icbc.PointSetBC(ob_xy_s, ob_vs, component=1)
        oc_T = dde.icbc.PointSetBC(ob_xy_s, ob_Ts, component=2)
        case.icbcocs += [oc_u, oc_v, oc_T]
        case.names["ICBCOCs"] += ["OC_u", "OC_v", "OC_T"]
    else:  # "none"
        ob_xy_s = np.empty([1, 2])
        n_ob = 0

    # ----------------------------------------------------------------------
    # define sampling points
    n_dmn = 2000
    n_bdr = 400
    data = dde.data.PDE(
        geom,
        pde,
        case.icbcocs,
        num_domain=n_dmn,
        num_boundary=n_bdr,
        train_distribution="pseudo",  # "Hammersley", "uniform", "pseudo"
        # anchors=ob_xy_s,
        # solution=func_sol,
        num_test=1000,
    )
    my_points = MyPoints(args=args, case=case, ob_xy_s=ob_xy_s, dtype=data.train_x.dtype)
    data = my_points.execute_modify(data=data)
    n_dmn, n_bdr = my_points.n_dmn, my_points.n_bdr
    n_bdr = 0 if bc_type == "none" else n_bdr

    # ----------------------------------------------------------------------
    # define maps (network and input/output transforms)
    maps = Maps(args=args, case=case, bc_type=bc_type)
    net = maps.net
    model = dde.Model(data, net)

    # ----------------------------------------------------------------------
    # define training and train
    output_dir = f"../results/{case_name}/uvT_{problem_type}/"
    output_dir += f"dmn{n_dmn}"
    if bc_type == "soft":
        output_dir += f"_bdr{n_bdr}"
    if oc_type == "soft":
        output_dir += f"_ob{n_ob}-N{efmt(args.noise_level)}"
    # output_dir += "_{:.1e}-{:.1e}_{:.1e}-{:.1e}/".format(scale_u, scale_v, scale_x, scale_y)

    output_dir += f"_u{efmt(scale_u)}-v{efmt(scale_v)}-T{efmt(scale_T)}" \
                  f"_x{efmt(scale_x)}-y{efmt(scale_y)}_x{efmt(shift_x)}-y{efmt(shift_y)}"
    if problem_type == "inverse_nu":
        output_dir += f"_nu{efmt(args.scale_nu)}"
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

    if problem_type == "inverse_nu":
        external_trainable_variables = [case.var_nu_s, ]
        variable_saver = dde.callbacks.VariableValue(
            external_trainable_variables, period=100, filename=output_dir + "vars_history_scaled.txt")
        callbacks.append(variable_saver)
    else:
        external_trainable_variables = []

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
    # restore a new model
    # model.restore(output_dir + "models/model_last-20000.pt")

    # ----------------------------------------------------------------------
    # post-process
    pp2d = PostProcessShear(args=args, case=case, model=model, output_dir=output_dir)
    pp2d.save_data()
    pp2d.save_metrics()
    pp2d.plot_save_loss_history()
    if problem_type == "inverse_nu":
        pp2d.save_var_metrics((case.nu, ), (case.var_nu_s / args.scale_nu, ), ("nu", ))
        pp2d.plot_save_var_history((case.nu, ), (args.scale_nu, ), (r"$\nu$", ), ("nu", ), ("m$^2$/s", ))
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
    parser.add_argument("--problem_type", type=str, default="forward", help="options: forward, inverse, inverse_nu")

    parser.add_argument("--scale_u", type=float, default=1.0)
    parser.add_argument("--scale_v", type=float, default=1.0)
    parser.add_argument("--scale_T", type=float, default=1.0)
    parser.add_argument("--scale_x", type=float, default=1.0)
    parser.add_argument("--scale_y", type=float, default=1.0)
    parser.add_argument("--shift_x", type=float, default=0.0)
    parser.add_argument("--shift_y", type=float, default=0.0)

    parser.add_argument("--noise_level", type=float, default=0.00,
                        help="noise level for observations, such as 0.02 (2%)")

    parser.add_argument("--nu_ini", type=float, default=1e-5)
    parser.add_argument("--scale_nu", type=float, default=1e4)

    parser.add_argument("--n_iter", type=int, default=20000)
    parser.add_argument("--i_run", type=int, default=1)

    # ----------------------------------------------------------------------
    # set arguments
    args = parser.parse_args()

    case_dict = {
        1: "jet_lam_plane",
        2: "jet_lam_round",
        3: "jet_tur_plane",
        4: "jet_tur_round",
    }
    para_dict = {
        "jet_lam_plane": (100, 1000, 0.1, 100, 100),
        "jet_lam_round": (100, 1000, 0.1, 100, 100),
        "jet_tur_plane": (1, 10, 0.1, 10, 100),
        "jet_tur_round": (1, 100, 0.1, 10, 100),
    }  # scale_u, scale_v, scale_T, scale_x, scale_y

    args.n_iter = 200
    # args.problem_type = "inverse"

    # ----------------------------------------------------------------------
    # run
    for case_id in (1,):
    # for case_id in (1, 2, 3, 4):
        args.case_name = case_dict[case_id]
        args.scale_u, args.scale_v, args.scale_T, args.scale_x, args.scale_y = para_dict[args.case_name]

        n_run = 1

        for args.i_run in range(1, 1 + n_run):
            output_dir = main(args)
        # cal_stat(output_dir[:-2], n_run)

        # for args.scale_x in (1, 10, 100, 1000):
        #     for args.scale_y in (5, 10, 100, 1000):
        #         if args.scale_x == 100 and args.scale_y == 100:
        #             continue
        #         else:
        #             for args.i_run in range(1, 1 + n_run):
        #                 output_dir = main(args)
        #             cal_stat(output_dir[:-2], n_run)

        # for args.shift_x in (0, -0.5):
        #     for args.shift_y in (0, -0.1):
        #         if args.shift_x == 0 and args.shift_y == 0:
        #             continue
        #         else:
        #             for args.i_run in range(1, 1 + n_run):
        #                 main(args)


