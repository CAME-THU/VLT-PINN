"""Network architecture, input transform, and output transform."""
# import deepxde as dde
import torch
from utils.fnn_modi import FNN


class Maps:
    def __init__(self, args, case, bc_type):
        self.args = args
        self.case = case

        self.net = FNN(
            layer_sizes=[2] + 6 * [50] + [3],
            # layer_sizes=[2] + [10, 20, 40] + 20 * [80] + [40, 20, 10] + [2],
            activation="tanh",  # "tanh", "sin"
            kernel_initializer="Glorot normal",
            input_transform=self.input_transform,
        )

        if bc_type == "soft":
            # self.net.apply_output_transform(self.output_physical_transform)
            # self.net.apply_output_transform(self.output_solution_transform)
            pass
        elif bc_type == "hard_LR":
            self.net.apply_output_transform(self.output_hardLR_transform)
        elif bc_type == "hard_DU":
            self.net.apply_output_transform(self.output_hardDU_transform)

    def input_transform(self, xy_s):
        xs, ys = xy_s[:, 0:1], xy_s[:, 1:2]
        xs_ = xs + self.case.delta * self.args.scale_x

        inputs = torch.cat([
            xs, ys,
            # 0.1 * xs, 0.2 * xs, 0.5 * xs, 2.0 * xs, 5.0 * xs, 10.0 * xs,
            # 0.1 * ys, 0.2 * ys, 0.5 * ys, 2.0 * ys, 5.0 * ys, 10.0 * ys,
            # xs_ ** (-1/3),
            # xs_ ** (-1/2),
            # xs_ ** (-2/3),
            # xs_ ** (-1),
            # ys * xs_ ** (-1/3),
            # ys * xs_ ** (-1/2),
            # ys * xs_ ** (-2/3),
            # ys * xs_ ** (-1),
            # eta, tanh_eta, tanh_eta ** 2,
            ], dim=1)
        return inputs

    def output_solution_transform(self, xy_s, uvT_s):
        us_ = self.case.func_us(xy_s)
        vs_ = self.case.func_vs(xy_s,)
        Ts_ = self.case.func_Ts(xy_s)
        return torch.cat([us_, vs_, Ts_], dim=1)

    def output_physical_transform(self, xy_s, uvT_s):
        # xs, ys = xy_s[:, 0:1], xy_s[:, 1:2]
        us, vs, Ts = uvT_s[:, 0:1], uvT_s[:, 1:2], uvT_s[:, 2:3]
        return torch.cat([us, vs, Ts], dim=1)

    def func_us(self, xs, ys):
        return self.case.func_us_ind(xs, ys)

    def func_vs(self, xs, ys):
        return self.case.func_vs_ind(xs, ys)

    def output_hardLR_transform(self, xy_s, uvT_s):
        xs, ys = xy_s[:, 0:1], xy_s[:, 1:2]
        us, vs, Ts = uvT_s[:, 0:1], uvT_s[:, 1:2], uvT_s[:, 2:3]

        scale_x, scale_y = self.args.scale_x, self.args.scale_y
        shift_x, shift_y = self.args.shift_x, self.args.shift_y
        xs_l, xs_r = scale_x * (self.case.x_l + shift_x), scale_x * (self.case.x_r + shift_x)
        # ys_l, ys_r = scale_y * (self.case.y_l + shift_y), scale_y * (self.case.y_r + shift_y)

        d_l, d_r = (xs - xs_l) / (xs_r - xs_l), (xs_r - xs) / (xs_r - xs_l)
        us_ = d_r * self.func_us(xs_l, ys) + d_l * self.func_us(xs_r, ys) + d_l * d_r * us
        vs_ = d_r * self.func_vs(xs_l, ys) + d_l * self.func_vs(xs_r, ys) + d_l * d_r * vs
        return torch.cat([us_, vs_, Ts], dim=1)

    def output_hardDU_transform(self, xy_s, uvT_s):
        xs, ys = xy_s[:, 0:1], xy_s[:, 1:2]
        us, vs, Ts = uvT_s[:, 0:1], uvT_s[:, 1:2], uvT_s[:, 2:3]

        scale_x, scale_y = self.args.scale_x, self.args.scale_y
        shift_x, shift_y = self.args.shift_x, self.args.shift_y
        # xs_l, xs_r = scale_x * (self.case.x_l + shift_x), scale_x * (self.case.x_r + shift_x)
        ys_l, ys_r = scale_y * (self.case.y_l + shift_y), scale_y * (self.case.y_r + shift_y)

        d_d, d_u = (ys - ys_l) / (ys_r - ys_l), (ys_r - ys) / (ys_r - ys_l)
        us_ = d_u * self.func_us(xs, ys_l) + d_d * self.func_us(xs, ys_r) + d_d * d_u * us
        vs_ = d_u * self.func_vs(xs, ys_l) + d_d * self.func_vs(xs, ys_r) + d_d * d_u * vs
        return torch.cat([us_, vs_, Ts], dim=1)


