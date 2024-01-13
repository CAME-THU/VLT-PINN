"""Case 6 (laminar axisymmetric wake): Parameters, reference solution, and governing equations."""
import math as mt
# import numpy as np
# import torch
import deepxde as dde


class Case:
    def __init__(self, args):
        self.args = args

        # ----------------------------------------------------------------------
        # define calculation domain size
        self.x_l, self.x_r = 0.0, 1.0
        self.y_l, self.y_r = 0.0, 0.2

        # define the names of independents, dependents, and equations
        self.names = {
            "independents": ["x", "y"],
            "dependents": ["u", "v"],
            "equations": ["continuity", "momentum_x"],
            "ICBCOCs": []}

        self.icbcocs = []  # initial, boundary, observation conditions

        self.var_nu_s = dde.Variable(args.nu_ini * args.scale_nu) if args.problem_type == "inverse_nu" else None

        # ----------------------------------------------------------------------
        # define parameters
        self.u_far, self.L, self.nu = 0.03, 0.05, 1.5e-5  # Re = 100
        # self.u_far, self.L, self.nu = 0.15, 0.1, 1.5e-5  # Re = 1000
        self.Re_L = self.u_far * self.L / self.nu

        self.c = 0.664 / mt.pi ** 0.5
        self.delta = 3 * self.L

    # ----------------------------------------------------------------------
    # theoretical solution with individual inputs
    def func_u_ind(self, x, y):
        x_ = x + self.delta
        f = mt.e ** (-0.25 * self.u_far / self.nu * y ** 2 / x_)
        return self.u_far * (1 - self.c * (self.L / x_) * f)

    def func_v_ind(self, x, y):
        x_ = x + self.delta
        f = mt.e ** (-0.25 * self.u_far / self.nu * y ** 2 / x_)
        return self.u_far * (-0.5) * self.c * (self.L * y / x_ ** 2) * f

    def func_psi_ind(self, x, y):
        x_ = x + self.delta
        f = mt.e ** (-0.25 * self.u_far / self.nu * y ** 2 / x_)
        return 0.5 * self.u_far * y ** 2 + 2 * self.c * self.nu * self.L * f

    # ----------------------------------------------------------------------
    # theoretical solution with stacked inputs
    def func_u(self, xy):
        return self.func_u_ind(xy[:, 0:1], xy[:, 1:2])

    def func_v(self, xy):
        return self.func_v_ind(xy[:, 0:1], xy[:, 1:2])

    def func_psi(self, xy):
        return self.func_psi_ind(xy[:, 0:1], xy[:, 1:2])

    # ----------------------------------------------------------------------
    # theoretical solution with linear-transformed inputs and outputs
    def func_us_ind(self, xs, ys):
        x = xs / self.args.scale_x - self.args.shift_x
        y = ys / self.args.scale_y - self.args.shift_y
        return self.func_u_ind(x, y) * self.args.scale_u

    def func_vs_ind(self, xs, ys):
        x = xs / self.args.scale_x - self.args.shift_x
        y = ys / self.args.scale_y - self.args.shift_y
        return self.func_v_ind(x, y) * self.args.scale_v

    def func_us(self, xy_s):
        return self.func_us_ind(xy_s[:, 0:1], xy_s[:, 1:2])

    def func_vs(self, xy_s):
        return self.func_vs_ind(xy_s[:, 0:1], xy_s[:, 1:2])

    # ----------------------------------------------------------------------
    # define pde
    def pde_uv(self, xy_s, uv_s):
        args = self.args
        xs, ys = xy_s[:, 0:1], xy_s[:, 1:2]
        us, vs = uv_s[:, 0:1], uv_s[:, 1:2]
        scale_u, scale_v = args.scale_u, args.scale_v
        scale_x, scale_y = args.scale_x, args.scale_y
        shift_x, shift_y = args.shift_x, args.shift_y

        us_xs = dde.grad.jacobian(uv_s, xy_s, i=0, j=0)
        us_ys = dde.grad.jacobian(uv_s, xy_s, i=0, j=1)
        # us_xsxs = dde.grad.hessian(uv_s, xy_s, component=0, i=0, j=0)
        us_ysys = dde.grad.hessian(uv_s, xy_s, component=0, i=1, j=1)

        # vs_xs = dde.grad.jacobian(uv_s, xy_s, i=1, j=0)
        vs_ys = dde.grad.jacobian(uv_s, xy_s, i=1, j=1)
        # vs_xsxs = dde.grad.hessian(uv_s, xy_s, component=1, i=0, j=0)
        # vs_ysys = dde.grad.hessian(uv_s, xy_s, component=1, i=1, j=1)

        cy = (scale_u / scale_v) * (scale_y / scale_x)
        cyy = scale_u * scale_y ** 2 / scale_x
        k = max(1.0, 1.0 / cy)
        nu = self.var_nu_s / args.scale_nu if args.problem_type == "inverse_nu" else self.nu
        ys_ = ys - scale_y * shift_y

        continuity = ys_ * (us_xs + cy * vs_ys) + cy * vs
        # momentum_x = ys_ * (us * us_xs + cy * vs * us_ys) - cyy * nu * (us_ys + ys_ * us_ysys)  # raw BL-Eq
        momentum_x = ys_ * ((self.u_far * scale_u) * us_xs) - cyy * nu * (us_ys + ys_ * us_ysys)  # wake-Eq
        continuity *= k  # / y_r
        momentum_x *= k  # / y_r

        return [continuity, momentum_x]


