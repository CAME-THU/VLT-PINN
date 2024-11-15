"""Network architecture, input transform, and output transform."""
import torch
from utils.networks import FNN


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

        self.net.apply_output_transform(self.output_denorm_transform)
        # self.net.apply_output_transform(self.output_solution_transform)

    def input_transform(self, xy):
        x, y = xy[:, 0:1], xy[:, 1:2]
        xs = (x + self.args.shifts["x"]) * self.args.scales["x"]
        ys = (y + self.args.shifts["y"]) * self.args.scales["y"]

        inputs = torch.cat([
            xs, ys,
            ], dim=1)
        return inputs

    def output_denorm_transform(self, xy, uvp_s):
        us, vs, ps = uvp_s[:, 0:1], uvp_s[:, 1:2], uvp_s[:, 2:3]
        u = us / self.args.scales["u"]
        v = vs / self.args.scales["v"]
        p = ps / self.args.scales["p"]
        return torch.cat([u, v, p], dim=1)

    def output_solution_transform(self, xy, uvp_s):
        u_sol = self.case.func_u_tensor(xy)
        v_sol = self.case.func_v_tensor(xy)
        p_sol = self.case.func_p_tensor(xy)
        return torch.cat([u_sol, v_sol, p_sol], dim=1)

