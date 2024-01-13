"""Case 1 (laminar plane jet): Parameters, reference solution, and governing equations."""
import math as mt
# import numpy as np
# import torch
import deepxde as dde


def tanh(x):
    """define my tanh so that it can be used for both numpy and tensor (i.e. avoid using np.tanh or torch.tanh)."""
    return (mt.e ** x - mt.e ** (-x)) / (mt.e ** x + mt.e ** (-x))


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
        self.delta = 5e-3
        self.K = 1e-5  # K = J / rho, kinetic momentum flux
        self.temp_flux = 1e-2
        self.Pr = 0.71
        self.beta = mt.gamma(self.Pr + 1) * mt.gamma(0.5) / mt.gamma(self.Pr + 1.5)

        eta_half_umax = mt.atanh(2 ** 0.5 * 0.5)
        width_half_umax = 2 * 3 * (4 / 3) ** (2 / 3) * (self.nu ** 2 / self.K) ** (1 / 3) * self.delta ** (2 / 3) * eta_half_umax
        umax = (2 / 3) * (3 / 4) ** (4 / 3) * (self.K ** 2 / (self.nu * self.delta)) ** (1 / 3)
        self.Re_width = umax * width_half_umax / self.nu

    def func_eta(self, x, y):
        c = (1 / 3) * (3 / 4) ** (2 / 3)
        return c * (self.K / self.nu ** 2) ** (1 / 3) * y * x ** (-2 / 3)

    # ----------------------------------------------------------------------
    # theoretical solution with individual inputs
    def func_u_ind(self, x, y):
        x_ = x + self.delta
        eta = self.func_eta(x_, y)
        tanh_eta = tanh(eta)
        c = (2 / 3) * (3 / 4) ** (4 / 3)
        u_max = c * (self.K ** 2 / (self.nu * x_)) ** (1 / 3)
        return u_max * (1 - tanh_eta ** 2)

    def func_v_ind(self, x, y):
        x_ = x + self.delta
        eta = self.func_eta(x_, y)
        tanh_eta = tanh(eta)
        c = (2 / 3) * (3 / 4) ** (2 / 3)
        v_far = c * (self.K * self.nu / x_ ** 2) ** (1 / 3)
        return v_far * (2 * eta * (1 - tanh_eta ** 2) - tanh_eta)

    def func_T_ind(self, x, y):
        x_ = x + self.delta
        eta = self.func_eta(x_, y)
        tanh_eta = tanh(eta)
        c = (0.5 / self.beta) * (4 / 3) ** (2 / 3)
        T_max = c * self.temp_flux * (self.K * self.nu * x_) ** (-1 / 3)
        return T_max * (1 - tanh_eta ** 2) ** self.Pr

    def func_psi_ind(self, x, y):
        x_ = x + self.delta
        eta = self.func_eta(x_, y)
        tanh_eta = tanh(eta)
        c = 2 * (3 / 4) ** (2 / 3)
        return c * (self.K * self.nu * x_) ** (1 / 3) * tanh_eta

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
        # xs, ys = xy_s[:, 0:1], xy_s[:, 1:2]
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

        continuity = us_xs + cy * vs_ys
        momentum_x = us * us_xs + cy * vs * us_ys - cyy * nu * us_ysys
        continuity *= k
        momentum_x *= k

        return [continuity, momentum_x]

    def pde_uvT(self, xy_s, uvT_s):
        args = self.args
        # xs, ys = xy_s[:, 0:1], xy_s[:, 1:2]
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

        continuity = us_xs + cy * vs_ys
        momentum_x = us * us_xs + cy * vs * us_ys - cyy * nu * us_ysys
        energy = us * Ts_xs + cy * vs * Ts_ys - cyy * (nu / self.Pr) * Ts_ysys
        continuity *= k
        momentum_x *= k
        energy *= k

        return [continuity, momentum_x, energy]
