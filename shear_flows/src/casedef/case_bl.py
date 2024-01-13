"""Case 8 (laminar flat-plate boundary layer): Parameters, reference solution, and governing equations."""
# import math as mt
import numpy as np
# import torch
import deepxde as dde
from scipy.interpolate import interp1d


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
        data = np.loadtxt("./casedef/Blasius.csv", delimiter=",")
        self.func_f = interp1d(data[:, 0], data[:, 1], kind="linear")
        self.func_f1 = interp1d(data[:, 0], data[:, 2], kind="linear")
        self.func_f2 = interp1d(data[:, 0], data[:, 3], kind="linear")

        self.u_far, self.nu = 0.15, 1.5e-5

        self.delta = 0.1  # 0.1, 1
        self.Re_x = self.u_far * self.delta / self.nu

    # ----------------------------------------------------------------------
    # theoretical solution with individual inputs
    def func_u_ind(self, x, y):
        x_ = x + self.delta
        eta = y * (self.u_far / (self.nu * x_)) ** 0.5
        return self.u_far * self.func_f1(eta)

    def func_v_ind(self, x, y):
        x_ = x + self.delta
        eta = y * (self.u_far / (self.nu * x_)) ** 0.5
        return 0.5 * (self.nu * self.u_far / x_) ** 0.5 * (eta * self.func_f1(eta) - self.func_f(eta))

    def func_psi_ind(self, x, y):
        x_ = x + self.delta
        eta = y * (self.u_far / (self.nu * x_)) ** 0.5
        return (self.nu * self.u_far * x_) ** 0.5 * self.func_f(eta)

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

