"""Case 1 (laminar plane jet): Parameters, reference solution, and governing equations."""
import math as mt
import numpy as np
# import torch
import deepxde as dde
from utils.icbcs import ScaledDirichletBC, ScaledNeumannBC, ScaledPointSetBC


def tanh(x):
    """define my tanh so that it can be used for both numpy and tensor (i.e. avoid using np.tanh or torch.tanh)."""
    return (mt.e ** x - mt.e ** (-x)) / (mt.e ** x + mt.e ** (-x))


class Case:
    def __init__(self, args):
        self.args = args

        # ----------------------------------------------------------------------
        # define calculation domain
        self.x_l, self.x_r = 0.0, 1.0
        self.y_l, self.y_r = 0.0, 0.2
        self.geom = dde.geometry.Rectangle(xmin=[self.x_l, self.y_l], xmax=[self.x_r, self.y_r])

        # define the names of independents, dependents, and equations
        self.names = {
            "independents": ["x", "y"],
            "dependents": ["u", "v"],
            "equations": ["continuity", "momentum_x"],
            "ICBCOCs": []}

        self.icbcocs = []  # initial, boundary, observation conditions

        self.nu_infe_s = dde.Variable(args.infer_paras["nu"] * args.scales["nu"]) if "nu" in args.infer_paras else None

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

        # ----------------------------------------------------------------------
        # define ICs, BCs, OCs
        self.define_icbcocs()

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
    # define pde
    def pde_uv(self, xy, uv):
        args = self.args
        # x, y = xy[:, 0:1], xy[:, 1:2]
        u, v = uv[:, 0:1], uv[:, 1:2]
        scale_u, scale_v = args.scales["u"], args.scales["v"]
        scale_x, scale_y = args.scales["x"], args.scales["y"]
        shift_x, shift_y = args.shifts["x"], args.shifts["y"]

        u_x = dde.grad.jacobian(uv, xy, i=0, j=0)
        u_y = dde.grad.jacobian(uv, xy, i=0, j=1)
        # u_xx = dde.grad.hessian(uv, xy, component=0, i=0, j=0)
        u_yy = dde.grad.hessian(uv, xy, component=0, i=1, j=1)

        # v_x = dde.grad.jacobian(uv, xy, i=1, j=0)
        v_y = dde.grad.jacobian(uv, xy, i=1, j=1)
        # v_xx = dde.grad.hessian(uv, xy, component=1, i=0, j=0)
        # v_yy = dde.grad.hessian(uv, xy, component=1, i=1, j=1)

        nu = self.nu_infe_s / args.scales["nu"] if "nu" in args.infer_paras else self.nu

        continuity = u_x + v_y
        momentum_x = u * u_x + v * u_y - nu * u_yy

        # cy = (scale_u / scale_v) * (scale_y / scale_x)
        # k = max(1.0, 1.0 / cy)
        # coef = k * (scale_u / scale_x)
        coef = max(scale_u / scale_x, scale_v / scale_y)
        continuity *= coef
        momentum_x *= coef * scale_u

        return [continuity, momentum_x]

    def pde_uvT(self, xy, uvT):
        args = self.args
        # x, y = xy[:, 0:1], xy[:, 1:2]
        u, v, T = uvT[:, 0:1], uvT[:, 1:2], uvT[:, 2:3]
        scale_u, scale_v, scale_T = args.scales["u"], args.scales["v"], args.scales["T"]
        scale_x, scale_y = args.scales["x"], args.scales["y"]
        shift_x, shift_y = args.shifts["x"], args.shifts["y"]

        u_x = dde.grad.jacobian(uvT, xy, i=0, j=0)
        u_y = dde.grad.jacobian(uvT, xy, i=0, j=1)
        # u_xx = dde.grad.hessian(uvT, xy, component=0, i=0, j=0)
        u_yy = dde.grad.hessian(uvT, xy, component=0, i=1, j=1)

        # v_x = dde.grad.jacobian(uvT, xy, i=1, j=0)
        v_y = dde.grad.jacobian(uvT, xy, i=1, j=1)
        # v_xx = dde.grad.hessian(uvT, xy, component=1, i=0, j=0)
        # v_yy = dde.grad.hessian(uvT, xy, component=1, i=1, j=1)

        T_x = dde.grad.jacobian(uvT, xy, i=2, j=0)
        T_y = dde.grad.jacobian(uvT, xy, i=2, j=1)
        # T_xx = dde.grad.hessian(uvT, xy, component=2, i=0, j=0)
        T_yy = dde.grad.hessian(uvT, xy, component=2, i=1, j=1)

        nu = self.nu_infe_s / args.scales["nu"] if "nu" in args.infer_paras else self.nu

        continuity = u_x + v_y
        momentum_x = u * u_x + v * u_y - nu * u_yy
        energy = u * T_x + v * T_y - (nu / self.Pr) * T_yy

        # cy = (scale_u / scale_v) * (scale_y / scale_x)
        # k = max(1.0, 1.0 / cy)
        # coef = k * (scale_u / scale_v)
        coef = max(scale_u / scale_x, scale_v / scale_y)
        continuity *= coef
        momentum_x *= coef * scale_u
        energy *= coef * scale_T

        return [continuity, momentum_x, energy]

    # ----------------------------------------------------------------------
    # define ICs, BCs, OCs
    def define_icbcocs(self):
        args = self.args
        geom = self.geom
        x_l, x_r = self.x_l, self.x_r
        y_l, y_r = self.y_l, self.y_r
        scale_u, scale_v = args.scales["u"], args.scales["v"]
        scale_x, scale_y = args.scales["x"], args.scales["y"]
        shift_x, shift_y = args.shifts["x"], args.shifts["y"]

        def bdr_down(xy, on_bdr):
            y = xy[1]
            return on_bdr and np.isclose(y, y_l)

        def bdr_left_right(xy, on_bdr):
            x = xy[0]
            return on_bdr and (np.isclose(x, x_l) or np.isclose(x, x_r))

        def bdr_down_up(xy, on_bdr):
            y = xy[1]
            return on_bdr and (np.isclose(y, y_l) or np.isclose(y, y_r))

        def bdr_left_right_up(xy, on_bdr):
            x, y = xy[0], xy[1]
            return on_bdr and (np.isclose(x, x_l) or np.isclose(x, x_r) or np.isclose(y, y_r))

        def func_dusdys(xy, uv, _):
            return dde.grad.jacobian(uv, xy, i=0, j=1) * scale_u / scale_y

        # bc_sym_dudy = ScaledNeumannBC(geom, lambda xy: 0, bdr_down, component=0, scale=scale_u / scale_y)  # unknown bug
        bc_sym_dudy = dde.icbc.OperatorBC(geom, func_dusdys, bdr_down)
        bc_sym_v = ScaledDirichletBC(geom, lambda xy: 0, bdr_down, component=1, scale=scale_v)

        # bc_all_u = ScaledDirichletBC(geom, self.func_u, lambda _, on_bdr: on_bdr, component=0, scale=scale_u)
        bc_all_v = ScaledDirichletBC(geom, self.func_v, lambda _, on_bdr: on_bdr, component=1, scale=scale_v)
        bc_left_right_u = ScaledDirichletBC(geom, self.func_u, bdr_left_right, component=0, scale=scale_u)
        bc_left_right_v = ScaledDirichletBC(geom, self.func_v, bdr_left_right, component=1, scale=scale_v)
        bc_down_up_u = ScaledDirichletBC(geom, self.func_u, bdr_down_up, component=0, scale=scale_u)
        bc_down_up_v = ScaledDirichletBC(geom, self.func_v, bdr_down_up, component=1, scale=scale_v)
        bc_left_right_up_u = ScaledDirichletBC(geom, self.func_u, bdr_left_right_up, component=0, scale=scale_u)
        bc_left_right_up_v = ScaledDirichletBC(geom, self.func_v, bdr_left_right_up, component=1, scale=scale_v)

        if args.bc_type == "soft":
            self.icbcocs += [bc_left_right_up_u, bc_sym_dudy, bc_all_v]
            self.names["ICBCOCs"] += ["BC_left_right_up_u", "BC_sym_dudy", "BC_all_v"]
            # self.icbcocs += [bc_left_right_up_u, bc_sym_dudy, bc_left_right_up_v, bc_sym_v]
            # self.names["ICBCOCs"] += ["BC_left_right_up_u", "BC_sym_dudy", "BC_left_right_up_v", "BC_sym_v"]
        elif args.bc_type == "hard_LR":
            self.icbcocs += [bc_down_up_u, bc_down_up_v]
            self.names["ICBCOCs"] += ["BC_down_up_u", "BC_down_up_v"]
        elif args.bc_type == "hard_DU":
            self.icbcocs += [bc_left_right_u, bc_left_right_v]
            self.names["ICBCOCs"] += ["BC_left_right_u", "BC_left_right_v"]
        else:  # "none"
            pass

        if args.oc_type == "soft":
            spc_ob = 0.02  # m
            # spc_ob = (y_r - y_l) / 20  # m
            n_ob_x, n_ob_y = int((self.x_r - self.x_l) / spc_ob) + 1, int((self.y_r - self.y_l) / spc_ob) + 1
            ob_xx, ob_yy = np.meshgrid(np.linspace(x_l, x_r, n_ob_x),
                                       np.linspace(y_l, y_r, n_ob_y), indexing="ij")
            ob_xy = np.vstack((np.ravel(ob_xx), np.ravel(ob_yy))).T
            n_ob = n_ob_x * n_ob_y

            ob_u = self.func_u(ob_xy)
            ob_v = self.func_v(ob_xy)
            normal_noise_u = np.random.randn(len(ob_u))[:, None]
            normal_noise_v = np.random.randn(len(ob_v))[:, None]
            ob_u += normal_noise_u * ob_u * args.noise_level
            ob_v += normal_noise_v * ob_v * args.noise_level

            oc_u = ScaledPointSetBC(ob_xy, ob_u, component=0, scale=scale_u)
            oc_v = ScaledPointSetBC(ob_xy, ob_v, component=1, scale=scale_v)
            self.icbcocs += [oc_u, oc_v]
            self.names["ICBCOCs"] += ["OC_u", "OC_v"]
        else:  # "none"
            ob_xy = np.empty([1, 2])
            n_ob = 0
        self.ob_xy = ob_xy
        self.n_ob = n_ob
