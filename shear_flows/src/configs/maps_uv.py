"""Network architecture, input transform, and output transform."""
# import deepxde as dde
import torch
from utils.networks import FNN


class Maps:
    def __init__(self, args, case):
        self.args = args
        self.case = case

        self.net = FNN(
            layer_sizes=[2] + 6 * [50] + [2],
            activation="tanh",  # "tanh", "sin"
            kernel_initializer="Glorot normal",
            input_transform=self.input_transform,
        )

        if args.bc_type == "hard_LR":
            self.net.apply_output_transform(self.output_hardLR_transform)
        elif args.bc_type == "hard_DU":
            self.net.apply_output_transform(self.output_hardDU_transform)
        else:  # "soft", "none"
            self.net.apply_output_transform(self.output_denorm_transform)
            # self.net.apply_output_transform(self.output_solution_transform)

    def input_transform(self, xy):
        x, y = xy[:, 0:1], xy[:, 1:2]
        # x_ = x + self.case.delta
        xs = (x + self.args.shifts["x"]) * self.args.scales["x"]
        ys = (y + self.args.shifts["y"]) * self.args.scales["y"]

        inputs = torch.cat([
            xs, ys,
            ], dim=1)
        return inputs

    def output_denorm_transform(self, xy, uv_s):
        us, vs = uv_s[:, 0:1], uv_s[:, 1:2]
        u = us / self.args.scales["u"]
        v = vs / self.args.scales["v"]
        return torch.cat([u, v], dim=1)

    def output_solution_transform(self, xy, uv_s):
        u_sol = self.case.func_u(xy)
        v_sol = self.case.func_v(xy)
        return torch.cat([u_sol, v_sol], dim=1)

    def func_u(self, x, y):
        return self.case.func_u_ind(x, y)

    def func_v(self, x, y):
        return self.case.func_v_ind(x, y)

    def output_hardLR_transform(self, xy, uv_s):
        x, y = xy[:, 0:1], xy[:, 1:2]
        us, vs = uv_s[:, 0:1], uv_s[:, 1:2]
        u = us / self.args.scales["u"]
        v = vs / self.args.scales["v"]
        x_l, x_r = self.case.x_l, self.case.x_r

        d_l, d_r = (x - x_l) / (x_r - x_l), (x_r - x) / (x_r - x_l)
        u_ = d_r * self.func_u(x_l, y) + d_l * self.func_u(x_r, y) + d_l * d_r * u
        v_ = d_r * self.func_v(x_l, y) + d_l * self.func_v(x_r, y) + d_l * d_r * v
        return torch.cat([u_, v_], dim=1)

    def output_hardDU_transform(self, xy, uv_s):
        x, y = xy[:, 0:1], xy[:, 1:2]
        us, vs = uv_s[:, 0:1], uv_s[:, 1:2]
        u = us / self.args.scales["u"]
        v = vs / self.args.scales["v"]
        y_l, y_r = self.case.y_l, self.case.y_r

        d_d, d_u = (y - y_l) / (y_r - y_l), (y_r - y) / (y_r - y_l)
        u_ = d_u * self.func_u(x, y_l) + d_d * self.func_u(x, y_r) + d_d * d_u * u
        v_ = d_u * self.func_v(x, y_l) + d_d * self.func_v(x, y_r) + d_d * d_u * v
        return torch.cat([u_, v_], dim=1)

