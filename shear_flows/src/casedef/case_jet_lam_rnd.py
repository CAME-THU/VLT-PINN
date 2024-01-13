"""Case 2 (laminar round jet): Parameters, reference solution, and governing equations."""
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
        self.nu = 1.5e-5  # nu = 1.7894e-5 / 1.225
        self.delta = 2e-2
        self.K = 1e-7  # K = J / rho, kinetic momentum flux
        self.temp_flux = 1e-4
        self.Pr = 0.71

        eta_half_umax = (2 ** 0.5 - 1) ** 0.5
        width_half_umax = 2 * 8 * (mt.pi / (3 * self.K)) ** 0.5 * self.nu * self.delta * eta_half_umax
        umax = 0.375 * self.K / (mt.pi * self.nu * self.delta)
        self.Re_width = umax * width_half_umax / self.nu

    def func_eta(self, x, y):
        return 0.125 * (3 * self.K / mt.pi) ** 0.5 / self.nu * (y / x)

    # ----------------------------------------------------------------------
    # theoretical solution with individual inputs
    def func_u_ind(self, x, y):
        x_ = x + self.delta
        eta = self.func_eta(x_, y)
        u_max = 0.375 * self.K / (mt.pi * self.nu * x_)
        return u_max * (1 + eta ** 2) ** (-2)

    def func_v_ind(self, x, y):
        x_ = x + self.delta
        eta = self.func_eta(x_, y)
        v_max_x4 = 0.5 * (3 * self.K / mt.pi) ** 0.5 * x_ ** (-1)
        return v_max_x4 * (eta - eta ** 3) * (1 + eta ** 2) ** (-2)

    def func_T_ind(self, x, y):
        x_ = x + self.delta
        eta = self.func_eta(x_, y)
        T_max = (1 + 2 * self.Pr) * self.temp_flux / (8 * mt.pi * self.nu * x_)
        return T_max * (1 + eta ** 2) ** (-2 * self.Pr)

    def func_psi_ind(self, x, y):
        x_ = x + self.delta
        eta = self.func_eta(x_, y)
        return self.nu * x_ * 4 * eta / (1 + eta ** 2)

    # ----------------------------------------------------------------------
    # theoretical solution with stacked inputs
    def func_u(self, xy):
        return self.func_u_ind(xy[:, 0:1], xy[:, 1:2])

    def func_v(self, xy):
        return self.func_v_ind(xy[:, 0:1], xy[:, 1:2])

    def func_T(self, xy):
        return self.func_T_ind(xy[:, 0:1], xy[:, 1:2])

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

    def func_Ts_ind(self, xs, ys):
        x = xs / self.args.scale_x - self.args.shift_x
        y = ys / self.args.scale_y - self.args.shift_y
        return self.func_T_ind(x, y) * self.args.scale_T

    def func_us(self, xy_s):
        return self.func_us_ind(xy_s[:, 0:1], xy_s[:, 1:2])

    def func_vs(self, xy_s):
        return self.func_vs_ind(xy_s[:, 0:1], xy_s[:, 1:2])

    def func_Ts(self, xy_s):
        return self.func_Ts_ind(xy_s[:, 0:1], xy_s[:, 1:2])

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
        momentum_x = ys_ * (us * us_xs + cy * vs * us_ys) - cyy * nu * (us_ys + ys_ * us_ysys)
        continuity *= k  # / y_r
        momentum_x *= k  # / y_r

        return [continuity, momentum_x]

    def pde_uvT(self, xy_s, uvT_s):
        args = self.args
        xs, ys = xy_s[:, 0:1], xy_s[:, 1:2]
        us, vs, Ts = uvT_s[:, 0:1], uvT_s[:, 1:2], uvT_s[:, 2:3]
        scale_u, scale_v, scale_T = args.scale_u, args.scale_v, args.scale_T
        scale_x, scale_y = args.scale_x, args.scale_y
        shift_x, shift_y = args.shift_x, args.shift_y

        us_xs = dde.grad.jacobian(uvT_s, xy_s, i=0, j=0)
        us_ys = dde.grad.jacobian(uvT_s, xy_s, i=0, j=1)
        # us_xsxs = dde.grad.hessian(uvT_s, xy_s, component=0, i=0, j=0)
        us_ysys = dde.grad.hessian(uvT_s, xy_s, component=0, i=1, j=1)

        # vs_xs = dde.grad.jacobian(uvT_s, xy_s, i=1, j=0)
        vs_ys = dde.grad.jacobian(uvT_s, xy_s, i=1, j=1)
        # vs_xsxs = dde.grad.hessian(uvT_s, xy_s, component=1, i=0, j=0)
        # vs_ysys = dde.grad.hessian(uvT_s, xy_s, component=1, i=1, j=1)

        Ts_xs = dde.grad.jacobian(uvT_s, xy_s, i=2, j=0)
        Ts_ys = dde.grad.jacobian(uvT_s, xy_s, i=2, j=1)
        # Ts_xsxs = dde.grad.hessian(uvT_s, xy_s, component=2, i=0, j=0)
        Ts_ysys = dde.grad.hessian(uvT_s, xy_s, component=2, i=1, j=1)

        cy = (scale_u / scale_v) * (scale_y / scale_x)
        cyy = scale_u * scale_y ** 2 / scale_x
        k = max(1.0, 1.0 / cy)
        nu = self.var_nu_s / args.scale_nu if args.problem_type == "inverse_nu" else self.nu
        ys_ = ys - scale_y * shift_y

        continuity = ys_ * (us_xs + cy * vs_ys) + cy * vs
        momentum_x = ys_ * (us * us_xs + cy * vs * us_ys) - cyy * nu * (us_ys + ys_ * us_ysys)
        energy = ys_ * (us * Ts_xs + cy * vs * Ts_ys) - cyy * (nu / self.Pr) * (Ts_ys + ys_ * Ts_ysys)
        continuity *= k  # / y_r
        momentum_x *= k  # / y_r
        energy *= k  # / y_r

        return [continuity, momentum_x, energy]

