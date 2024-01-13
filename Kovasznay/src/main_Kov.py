"""PINN to solve extended Kovasznay flows."""
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
from configs.points_Kov import MyPoints
from configs.maps_uvp import Maps
from configs.post_Kov import PostProcessKov
from utils.utils import efmt, cal_stat


def main(args):
    case = Case(args)
    case_name = f"Re{case.Re}_a{case.a}_b{case.b}_m{case.m}_{case.x_l}_{case.x_r}_{case.y_l}_{case.y_r}"

    # ----------------------------------------------------------------------
    # define constants
    problem_type = args.problem_type  # "forward", "inverse", "inverse_nu"
    if problem_type == "forward":
        bc_type = "soft"  # "none", "soft"
        oc_type = "none"  # "none", "soft"
    else:
        bc_type = "none"  # "none", "soft"
        oc_type = "soft"  # "none", "soft"

    scale_u, scale_v, scale_p = args.scale_u, args.scale_v, args.scale_p
    scale_x, scale_y = args.scale_x, args.scale_y
    shift_x, shift_y = args.shift_x, args.shift_y

    x_l, x_r, y_l, y_r = case.x_l, case.x_r, case.y_l, case.y_r
    xs_l, xs_r = scale_x * (x_l + shift_x), scale_x * (x_r + shift_x)
    ys_l, ys_r = scale_y * (y_l + shift_y), scale_y * (y_r + shift_y)

    # ----------------------------------------------------------------------
    # define pde
    pde = case.pde

    # ----------------------------------------------------------------------
    # define calculation domain (geometry and boundaries), ICs, BCs, OCs
    geom = dde.geometry.Rectangle(xmin=[xs_l, ys_l], xmax=[xs_r, ys_r])

    def bdr_outflow(xy_s, on_boundary):
        return on_boundary and np.isclose(xy_s[0], xs_r)

    bc_u = dde.icbc.DirichletBC(geom, case.func_us, lambda _, on_boundary: on_boundary, component=0)
    bc_v = dde.icbc.DirichletBC(geom, case.func_vs, lambda _, on_boundary: on_boundary, component=1)
    bc_right_p = dde.icbc.DirichletBC(geom, case.func_ps, bdr_outflow, component=2)

    if bc_type == "soft":
        case.icbcocs += [bc_u, bc_v, bc_right_p]
        case.names["ICBCOCs"] += ["BC_u", "BC_v", "BC_right_p"]
    else:  # "none"
        pass

    if oc_type == "soft":
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

        oc_u = dde.icbc.PointSetBC(ob_xy_s, ob_us, component=0)
        oc_v = dde.icbc.PointSetBC(ob_xy_s, ob_vs, component=1)
        oc_p = dde.icbc.PointSetBC(ob_xy_s, ob_ps, component=2)
        case.icbcocs += [oc_u, oc_v, oc_p]
        case.names["ICBCOCs"] += ["OC_u", "OC_v", "OC_p"]
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
    maps = Maps(args=args, case=case)
    net = maps.net
    model = dde.Model(data, net)

    # ----------------------------------------------------------------------
    # define training and train
    output_dir = f"../results/{case_name}/{problem_type}/"
    output_dir += f"dmn{n_dmn}"
    if bc_type == "soft":
        output_dir += f"_bdr{n_bdr}"
    if oc_type == "soft":
        output_dir += f"_ob{n_ob}-N{efmt(args.noise_level)}"
    # output_dir += "_{:.1e}-{:.1e}_{:.1e}-{:.1e}/".format(scale_u, scale_v, scale_x, scale_y)

    output_dir += f"_u{efmt(scale_u)}-v{efmt(scale_v)}-p{efmt(scale_p)}" \
                  f"_x{efmt(scale_x)}-y{efmt(scale_y)}_x{efmt(shift_x)}-y{efmt(shift_y)}"
    if problem_type == "inverse_nu":
        # output_dir += f"_nu{efmt(args.scale_nu)}"
        output_dir += f"_nu1.0e-3-{efmt(args.scale_nu)}"
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
                model_save_path=output_dir + "models/model_last", )

    # ----------------------------------------------------------------------
    # restore a new model
    # model.restore(output_dir + "models/model_last-20000.pt")

    # ----------------------------------------------------------------------
    # post-process
    pp2d = PostProcessKov(args=args, case=case, model=model, output_dir=output_dir)
    pp2d.save_data()
    pp2d.save_metrics()
    pp2d.plot_save_loss_history()
    if problem_type == "inverse_nu":
        pp2d.save_var_metrics((case.nu, ), (case.var_nu_s / args.scale_nu, ), ("nu", ))
        pp2d.plot_save_var_history((case.nu, ), (args.scale_nu, ), (r"$\nu$", ), ("nu", ), ("m$^2$/s", ))
    pp2d.delete_old_models()
    pp2d.plot_sampling_points(figsize=(4, 8))
    pp2d.plot_2dfields(figsize=(4, 8))
    pp2d.plot_2dfields_comp(figsize=(5, 8))
    pp2d.plot_1dcurves(select_x=(x_l + 0.0 * (x_r - x_l), x_l + 0.95 * (x_r - x_l), x_l + 1.0 * (x_r - x_l)),
                       select_y=(y_l + 0.0 * (y_r - y_l), y_l + 0.50 * (y_r - y_l), y_l + 1.0 * (y_r - y_l)))
    # if type(net) == dde.nn.pytorch.fnn.FNN:
    #     pp2d.plot_lossgrads_fnn_uv()

    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyper-parameters")
    parser.add_argument("--case_id", type=int, default=1)
    parser.add_argument("--problem_type", type=str, default="forward", help="options: forward, inverse, inverse_nu")

    parser.add_argument("--scale_u", type=float, default=1.0)
    parser.add_argument("--scale_v", type=float, default=1.0)
    parser.add_argument("--scale_p", type=float, default=1.0)
    parser.add_argument("--scale_x", type=float, default=1.0)
    parser.add_argument("--scale_y", type=float, default=1.0)
    parser.add_argument("--shift_x", type=float, default=0.0)
    parser.add_argument("--shift_y", type=float, default=0.0)

    parser.add_argument("--noise_level", type=float, default=0.00,
                        help="noise level for observations, such as 0.02 (2%)")

    parser.add_argument("--nu_ini", type=float, default=1e-3)  # 4e-3, 1e-2
    parser.add_argument("--scale_nu", type=float, default=1e2)

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
    # args.problem_type = "inverse"

    # ----------------------------------------------------------------------
    # run
    for args.case_id in (1, ):
    # for args.case_id in (1, 2, ):
        args.scale_u, args.scale_v, args.scale_p, args.scale_x, args.scale_y, \
        args.shift_x, args.shift_y = para_dict[args.case_id]

        n_run = 1
        for args.i_run in range(1, 1 + n_run):
            output_dir = main(args)
        # cal_stat(output_dir[:-2], n_run)

        # # for args.problem_type in ("inverse", ):
        # for args.problem_type in ("inverse_nu",):
        # # for args.problem_type in ("inverse", "inverse_nu"):
        #     for args.noise_level in (0.00, 0.02):
        #         if args.case_id == 1 and args.noise_level == 0.0:
        #             continue
        #         else:
        #             for args.i_run in range(1, 1 + n_run):
        #                 args.nu_s = dde.Variable(nu_0 * args.scale_nu)
        #                 output_dir = main(args)
        #             cal_stat(output_dir[:-2], n_run)

        # args.problem_type = "inverse_nu"
        # # for args.noise_level in (0.00, 0.02):
        # for args.noise_level in (0.00, ):
        #     # for args.scale_nu in (1e0, 1e1, 1e3, 1e4):
        #     # for args.scale_nu in (1e-3, 1e-2, 1e-1, 1e5):
        #     for args.scale_nu in (1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5):
        #         for args.i_run in range(1, 1 + n_run):
        #             args.nu_s = dde.Variable(nu_0 * args.scale_nu)
        #             main(args)

        # for args.shift_x in (0, -0.2, ):
        #     for args.i_run in range(1, 1 + n_run):
        #         main(args)

        # for args.shift_x in (0, -0.2):
        #     for args.shift_y in (0, -0.5):
        #         for args.i_run in range(1, 1 + n_run):
        #             main(args)

        # for args.scale_x in (1, 10, ):
        #     for args.shift_x in (0, 0.2, ):
        #         for args.i_run in range(1, 1 + n_run):
        #             main(args)

        # for args.scale_x in (5, 10, 100, 1000):
        #     for args.scale_y in (0.1, 1, 10, 100):
        #         for args.i_run in range(1, 1 + n_run):
        #             main(args)
