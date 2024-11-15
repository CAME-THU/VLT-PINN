"""Case parameters, reference solutions, governing equations, IC/BC/OCs."""
import math as mt
import numpy as np
import torch
import deepxde as dde
from utils.icbcs import ScaledDirichletBC, ScaledPointSetBC


class Case:
    def __init__(self, args):
        self.args = args

        # ----------------------------------------------------------------------
        # define calculation domain
        if args.case_id == 1:
            self.x_l, self.x_r = -0.2, 0.0
        elif args.case_id == 2:
            self.x_l, self.x_r = 0.0, 0.2
        else:
            self.x_l, self.x_r = None, None
        self.y_l, self.y_r = 0.0, 1.0
        self.geom = dde.geometry.Rectangle(xmin=[self.x_l, self.y_l], xmax=[self.x_r, self.y_r])

        # define the names of independents, dependents, and equations
        self.names = {
            "independents": ["x", "y"],
            "dependents": ["u", "v", "p"],
            "equations": ["continuity", "momentum_x", "momentum_y"],
            "ICBCOCs": []}

        self.icbcocs = []  # initial, boundary, observation conditions

        self.nu_infe_s = dde.Variable(args.infer_paras["nu"] * args.scales["nu"]) if "nu" in args.infer_paras else None

        # ----------------------------------------------------------------------
        # define parameters
        if args.case_id == 1:
            self.a, self.b, self.m, self.nu = 1, 0.25, 0, 0.005
        elif args.case_id == 2:
            self.a, self.b, self.m, self.nu = 1e-17, 0.25, 0, 0.005
        else:
            self.a, self.b, self.m, self.nu = None, None, None, None
        self.Re = 1 / self.nu
        self.ksi = self.Re / 2 + (-1) ** self.m * mt.sqrt(0.25 * self.Re ** 2 + 4 * mt.pi ** 2 * self.b ** 2)
        
        # ----------------------------------------------------------------------
        # define ICs, BCs, OCs
        self.define_icbcocs()

    # ----------------------------------------------------------------------
    # theoretical solution with individual inputs
    def func_u_ind(self, x, y):
        a, b, ksi = self.a, self.b, self.ksi
        return 1 - a * np.exp(ksi * x) * np.cos(2 * np.pi * b * y)

    def func_v_ind(self, x, y):
        a, b, ksi = self.a, self.b, self.ksi
        return a * ksi / (2 * np.pi * b) * np.exp(ksi * x) * np.sin(2 * np.pi * b * y)

    def func_p_ind(self, x, y):
        a, b, ksi = self.a, self.b, self.ksi
        return a ** 2 / 2 * (1 - np.exp(2 * ksi * x))

    def func_u_ind_tensor(self, x, y):
        a, b, ksi = self.a, self.b, self.ksi
        return 1 - a * torch.exp(ksi * x) * torch.cos(2 * torch.pi * b * y)

    def func_v_ind_tensor(self, x, y):
        a, b, ksi = self.a, self.b, self.ksi
        return a * ksi / (2 * torch.pi * b) * torch.exp(ksi * x) * torch.sin(2 * torch.pi * b * y)

    def func_p_ind_tensor(self, x, y):
        a, b, ksi = self.a, self.b, self.ksi
        return a ** 2 / 2 * (1 - torch.exp(2 * ksi * x))

    # ----------------------------------------------------------------------
    # theoretical solution with stacked inputs
    def func_u(self, xy):
        return self.func_u_ind(xy[:, 0:1], xy[:, 1:2])

    def func_v(self, xy):
        return self.func_v_ind(xy[:, 0:1], xy[:, 1:2])

    def func_p(self, xy):
        return self.func_p_ind(xy[:, 0:1], xy[:, 1:2])

    def func_u_tensor(self, xy):
        return self.func_u_ind_tensor(xy[:, 0:1], xy[:, 1:2])

    def func_v_tensor(self, xy):
        return self.func_v_ind_tensor(xy[:, 0:1], xy[:, 1:2])

    def func_p_tensor(self, xy):
        return self.func_p_ind_tensor(xy[:, 0:1], xy[:, 1:2])

    # ----------------------------------------------------------------------
    # define pde
    def pde(self, xy, uvp):
        args = self.args
        u, v, p = uvp[:, 0:1], uvp[:, 1:2], uvp[:, 2:]
        scale_u, scale_v, scale_p = args.scales["u"], args.scales["v"], args.scales["p"]
        scale_x, scale_y = args.scales["x"], args.scales["y"]
        # shift_x, shift_y = args.shifts["x"], args.shifts["y"]

        u_x = dde.grad.jacobian(uvp, xy, i=0, j=0)
        u_y = dde.grad.jacobian(uvp, xy, i=0, j=1)
        u_xx = dde.grad.hessian(uvp, xy, component=0, i=0, j=0)
        u_yy = dde.grad.hessian(uvp, xy, component=0, i=1, j=1)

        v_x = dde.grad.jacobian(uvp, xy, i=1, j=0)
        v_y = dde.grad.jacobian(uvp, xy, i=1, j=1)
        v_xx = dde.grad.hessian(uvp, xy, component=1, i=0, j=0)
        v_yy = dde.grad.hessian(uvp, xy, component=1, i=1, j=1)

        p_x = dde.grad.jacobian(uvp, xy, i=2, j=0)
        p_y = dde.grad.jacobian(uvp, xy, i=2, j=1)

        nu = self.nu_infe_s / args.scales["nu"] if "nu" in args.infer_paras else self.nu

        continuity = u_x + v_y
        momentum_x = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
        momentum_y = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)

        # cy = (scale_u / scale_v) * (scale_y / scale_x)
        # k = max(1.0, 1.0 / cy)
        # coef = k * (scale_u / scale_x)
        coef = max(scale_u / scale_x, scale_v / scale_y)
        continuity *= coef
        momentum_x *= coef * scale_u
        momentum_y *= coef * scale_v

        return [continuity, momentum_x, momentum_y]

    # ----------------------------------------------------------------------
    # define ICs, BCs, OCs
    def define_icbcocs(self):
        args = self.args
        geom = self.geom
        x_l, x_r = self.x_l, self.x_r
        y_l, y_r = self.y_l, self.y_r
        scale_u, scale_v, scale_p = args.scales["u"], args.scales["v"], args.scales["p"]

        def bdr_outflow(xy, on_bdr):
            return on_bdr and np.isclose(xy[0], x_r)

        bc_u = ScaledDirichletBC(geom, self.func_u, lambda _, on_bdr: on_bdr, component=0, scale=scale_u)
        bc_v = ScaledDirichletBC(geom, self.func_v, lambda _, on_bdr: on_bdr, component=1, scale=scale_v)
        bc_right_p = ScaledDirichletBC(geom, self.func_p, bdr_outflow, component=2, scale=scale_p)

        if args.bc_type == "soft":
            self.icbcocs += [bc_u, bc_v, bc_right_p]
            self.names["ICBCOCs"] += ["BC_u", "BC_v", "BC_right_p"]
        else:  # "none"
            pass

        if args.oc_type == "soft":
            spc_ob = 0.02  # m
            # spc_ob = (self.x_r - self.x_l) / 20  # m
            n_ob_x, n_ob_y = int((self.x_r - self.x_l) / spc_ob) + 1, int((self.y_r - self.y_l) / spc_ob) + 1
            ob_xx, ob_yy = np.meshgrid(np.linspace(x_l, x_r, n_ob_x),
                                           np.linspace(y_l, y_r, n_ob_y), indexing="ij")
            ob_xy = np.vstack((np.ravel(ob_xx), np.ravel(ob_yy))).T
            n_ob = n_ob_x * n_ob_y

            ob_u = self.func_u(ob_xy)
            ob_v = self.func_v(ob_xy)
            ob_p = self.func_p(ob_xy)
            normal_noise_u = np.random.randn(len(ob_u))[:, None]
            normal_noise_v = np.random.randn(len(ob_v))[:, None]
            normal_noise_p = np.random.randn(len(ob_p))[:, None]
            ob_u += normal_noise_u * ob_u * args.noise_level
            ob_v += normal_noise_v * ob_v * args.noise_level
            ob_p += normal_noise_p * ob_p * args.noise_level

            oc_u = ScaledPointSetBC(ob_xy, ob_u, component=0, scale=scale_u)
            oc_v = ScaledPointSetBC(ob_xy, ob_v, component=1, scale=scale_v)
            oc_p = ScaledPointSetBC(ob_xy, ob_p, component=2, scale=scale_p)
            self.icbcocs += [oc_u, oc_v, oc_p]
            self.names["ICBCOCs"] += ["OC_u", "OC_v", "OC_p"]
        else:  # "none"
            ob_xy = np.empty([1, 2])
            n_ob = 0
        self.ob_xy = ob_xy
        self.n_ob = n_ob

