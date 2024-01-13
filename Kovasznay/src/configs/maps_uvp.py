"""Network architecture, input transform, and output transform."""
import torch
from utils.fnn_modi import FNN


class Maps:
    def __init__(self, args, case):
        self.args = args
        self.case = case

        self.net = FNN(
            layer_sizes=[2] + 6 * [50] + [3],
            # layer_sizes=[2] + [10, 20, 40] + 20 * [80] + [40, 20, 10] + [2],
            activation="tanh",  # "tanh", "sin"
            kernel_initializer="Glorot normal",
            input_transform=self.input_transform,
        )

        # self.net.apply_output_transform(self.output_solution_transform)
        # self.net.apply_output_transform(self.output_physical_transform)

    def input_transform(self, xy_s):
        xs, ys = xy_s[:, 0:1], xy_s[:, 1:2]

        inputs = torch.cat([
            xs, ys,
            ], dim=1)
        return inputs

    def output_solution_transform(self, xy_s, uvp_s):
        us_ = self.case.func_us_tensor(xy_s)
        vs_ = self.case.func_vs_tensor(xy_s)
        ps_ = self.case.func_ps_tensor(xy_s)
        return torch.cat([us_, vs_, ps_], dim=1)

    def output_physical_transform(self, xy_s, uvp_s):
        # xs, ys = xy_s[:, 0:1], xy_s[:, 1:2]
        us, vs, ps = uvp_s[:, 0:1], uvp_s[:, 1:2], uvp_s[:, 2:3]
        return torch.cat([us, vs, ps], dim=1)

