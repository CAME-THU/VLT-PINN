"""Case parameters, reference solutions, and governing equations."""
import math as mt
import numpy as np
import torch
import deepxde as dde


class Case:
    def __init__(self, args):
        self.args = args

        # ----------------------------------------------------------------------
        # define calculation domain size
        if args.case_id == 1:
            self.x_l, self.x_r = -0.2, 0.0
        elif args.case_id == 2:
            self.x_l, self.x_r = 0.0, 0.2
        else:
            self.x_l, self.x_r = None, None
        self.y_l, self.y_r = 0.0, 1.0

        # define the names of independents, dependents, and equations
        self.names = {
            "independents": ["x", "y"],
            "dependents": ["u", "v", "p"],
            "equations": ["continuity", "momentum_x", "momentum_y"],
            "ICBCOCs": []}

        self.icbcocs = []  # initial, boundary, observation conditions

        self.var_nu_s = dde.Variable(args.nu_ini * args.scale_nu) if args.problem_type == "inverse_nu" else None

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

    def func_ps_ind(self, xs, ys):
        x = xs / self.args.scale_x - self.args.shift_x
        y = ys / self.args.scale_y - self.args.shift_y
        return self.func_p_ind(x, y) * self.args.scale_p

    def func_us(self, xy_s):
        return self.func_us_ind(xy_s[:, 0:1], xy_s[:, 1:2])

    def func_vs(self, xy_s):
        return self.func_vs_ind(xy_s[:, 0:1], xy_s[:, 1:2])

    def func_ps(self, xy_s):
        return self.func_ps_ind(xy_s[:, 0:1], xy_s[:, 1:2])

    def func_us_tensor(self, xy_s):
        xs, ys = xy_s[:, 0:1], xy_s[:, 1:2]
        x = xs / self.args.scale_x - self.args.shift_x
        y = ys / self.args.scale_y - self.args.shift_y
        return self.func_u_ind_tensor(x, y) * self.args.scale_u

    def func_vs_tensor(self, xy_s):
        xs, ys = xy_s[:, 0:1], xy_s[:, 1:2]
        x = xs / self.args.scale_x - self.args.shift_x
        y = ys / self.args.scale_y - self.args.shift_y
        return self.func_v_ind_tensor(x, y) * self.args.scale_v

    def func_ps_tensor(self, xy_s):
        xs, ys = xy_s[:, 0:1], xy_s[:, 1:2]
        x = xs / self.args.scale_x - self.args.shift_x
        y = ys / self.args.scale_y - self.args.shift_y
        return self.func_p_ind_tensor(x, y) * self.args.scale_p

    # ----------------------------------------------------------------------
    # define pde
    def pde(self, xy_s, uvp_s):
        args = self.args
        us, vs, ps = uvp_s[:, 0:1], uvp_s[:, 1:2], uvp_s[:, 2:]
        scale_u, scale_v, scale_p = args.scale_u, args.scale_v, args.scale_p
        scale_x, scale_y = args.scale_x, args.scale_y
        # shift_x, shift_y = args.shift_x, args.shift_y

        us_xs = dde.grad.jacobian(uvp_s, xy_s, i=0, j=0)
        us_ys = dde.grad.jacobian(uvp_s, xy_s, i=0, j=1)
        us_xsxs = dde.grad.hessian(uvp_s, xy_s, component=0, i=0, j=0)
        us_ysys = dde.grad.hessian(uvp_s, xy_s, component=0, i=1, j=1)

        vs_xs = dde.grad.jacobian(uvp_s, xy_s, i=1, j=0)
        vs_ys = dde.grad.jacobian(uvp_s, xy_s, i=1, j=1)
        vs_xsxs = dde.grad.hessian(uvp_s, xy_s, component=1, i=0, j=0)
        vs_ysys = dde.grad.hessian(uvp_s, xy_s, component=1, i=1, j=1)

        ps_xs = dde.grad.jacobian(uvp_s, xy_s, i=2, j=0)
        ps_ys = dde.grad.jacobian(uvp_s, xy_s, i=2, j=1)

        cy = (scale_u / scale_v) * (scale_y / scale_x)
        cxx = scale_u * scale_x
        cyy = scale_u * scale_y ** 2 / scale_x
        cpx = scale_u ** 2 / scale_p
        cpy = (scale_u * scale_v / scale_p) * (scale_y / scale_x)
        # k = max(1.0, 1.0 / cy, 1.0 / cxx, 1.0 / cyy, 1.0 / cpx, 1.0 / cpy)
        k = max(1.0, 1.0 / cy)
        nu = self.var_nu_s / args.scale_nu if args.problem_type == "inverse_nu" else self.nu

        continuity = us_xs + cy * vs_ys
        momentum_x = us * us_xs + cy * vs * us_ys + cpx * ps_xs - nu * (cxx * us_xsxs + cyy * us_ysys)
        momentum_y = us * vs_xs + cy * vs * vs_ys + cpy * ps_ys - nu * (cxx * vs_xsxs + cyy * vs_ysys)
        continuity *= k
        momentum_x *= k
        momentum_y *= k

        return [continuity, momentum_x, momentum_y]

