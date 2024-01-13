"""Define the PDE and BC residual points."""
import numpy as np
from utils.utils import my_logspace


class MyPoints:
    def __init__(self, args, case, ob_xy_s, dtype):
        self.case = case
        self.n_dmn = 0
        self.n_bdr = 0

        scale_x, scale_y = args.scale_x, args.scale_y
        shift_x, shift_y = args.shift_x, args.shift_y
        x_l, x_r, y_l, y_r = case.x_l, case.x_r, case.y_l, case.y_r
        xs_l, xs_r = scale_x * (x_l + shift_x), scale_x * (x_r + shift_x)
        ys_l, ys_r = scale_y * (y_l + shift_y), scale_y * (y_r + shift_y)

        spc_base = (y_r - y_l) / 200

        exp_base = 10
        n_dmn_x, n_dmn_y = 60, 300
        n_bdr_x, n_bdr_y = int((x_r - x_l) / spc_base) + 1, int((y_r - y_l) / spc_base) + 1

        if case.m == 0:
            xs_dmn = my_logspace(xs_l, xs_r, n_dmn_x, exp_base, "end")
            xs_bdr = my_logspace(xs_l, xs_r, n_bdr_x, exp_base, "end")
        else:
            xs_dmn = my_logspace(xs_l, xs_r, n_dmn_x, exp_base, "start")
            xs_bdr = my_logspace(xs_l, xs_r, n_bdr_x, exp_base, "start")

        ys_dmn = np.linspace(ys_l, ys_r, n_dmn_y)
        ys_bdr = np.linspace(ys_l, ys_r, n_bdr_y)

        # modify sampling points in domain
        xsxs_dmn, ysys_dmn = np.meshgrid(xs_dmn, ys_dmn, indexing="ij")
        self.xy_s_dmn = np.vstack((np.ravel(xsxs_dmn), np.ravel(ysys_dmn))).T
        self.n_dmn = n_dmn_x * n_dmn_y

        # modify sampling points on boundary
        xy_s_l = np.vstack([np.ones_like(ys_bdr) * xs_l, ys_bdr]).astype(dtype).T
        xy_s_r = np.vstack([np.ones_like(ys_bdr) * xs_r, ys_bdr]).astype(dtype).T
        xy_s_d = np.vstack([xs_bdr, np.ones_like(xs_bdr) * ys_l]).astype(dtype).T
        xy_s_u = np.vstack([xs_bdr, np.ones_like(xs_bdr) * ys_r]).astype(dtype).T

        self.n_bdr = len(xy_s_l) + len(xy_s_r) + len(xy_s_d) + len(xy_s_u)

        self.bc_data_dict = {
            "BC_u": np.vstack([xy_s_l, xy_s_r, xy_s_d, xy_s_u]),
            "BC_v": np.vstack([xy_s_l, xy_s_r, xy_s_d, xy_s_u]),
            "BC_right_p": xy_s_r,

            "OC_u": ob_xy_s.astype(dtype),
            "OC_v": ob_xy_s.astype(dtype),
            "OC_p": ob_xy_s.astype(dtype),
        }

        self.xy_s_dmn_test = np.vstack([np.random.rand(10) * (xs_r - xs_l) + xs_l,
                                        np.random.rand(10) * (ys_r - ys_l) + ys_l]).astype(dtype).T

    def execute_modify(self, data):
        data.replace_with_anchors(self.xy_s_dmn)

        if len(self.case.names["ICBCOCs"]) != 0:
            data.train_x_bc = np.vstack([self.bc_data_dict[key] for key in self.case.names["ICBCOCs"]])
        data.num_bcs = [len(self.bc_data_dict[key]) for key in self.case.names["ICBCOCs"]]

        # self.n_bc = sum(data.num_bcs)
        data.train_x = np.vstack([data.train_x_bc, data.train_x_all])
        data.test_x = np.vstack([data.train_x_bc, self.xy_s_dmn_test])

        return data

