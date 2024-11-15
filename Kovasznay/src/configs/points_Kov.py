"""Define the PDE and BC residual points."""
import numpy as np
from utils.utils import my_logspace


class MyPoints:
    def __init__(self, args, case, ob_xy, dtype):
        self.case = case
        self.n_dmn = 0
        self.n_bdr = 0

        x_l, x_r, y_l, y_r = case.x_l, case.x_r, case.y_l, case.y_r

        spc_base = (y_r - y_l) / 200

        exp_base = 10
        n_dmn_x, n_dmn_y = 60, 300
        n_bdr_x, n_bdr_y = int((x_r - x_l) / spc_base) + 1, int((y_r - y_l) / spc_base) + 1

        if case.m == 0:
            x_dmn = my_logspace(x_l, x_r, n_dmn_x, exp_base, "end")
            x_bdr = my_logspace(x_l, x_r, n_bdr_x, exp_base, "end")
        else:
            x_dmn = my_logspace(x_l, x_r, n_dmn_x, exp_base, "start")
            x_bdr = my_logspace(x_l, x_r, n_bdr_x, exp_base, "start")

        y_dmn = np.linspace(y_l, y_r, n_dmn_y)
        y_bdr = np.linspace(y_l, y_r, n_bdr_y)

        # modify sampling points in domain
        xx_dmn, yy_dmn = np.meshgrid(x_dmn, y_dmn, indexing="ij")
        self.xy_dmn = np.vstack((np.ravel(xx_dmn), np.ravel(yy_dmn))).T
        self.n_dmn = n_dmn_x * n_dmn_y

        # modify sampling points on boundary
        xy_l = np.vstack([np.ones_like(y_bdr) * x_l, y_bdr]).astype(dtype).T
        xy_r = np.vstack([np.ones_like(y_bdr) * x_r, y_bdr]).astype(dtype).T
        xy_d = np.vstack([x_bdr, np.ones_like(x_bdr) * y_l]).astype(dtype).T
        xy_u = np.vstack([x_bdr, np.ones_like(x_bdr) * y_r]).astype(dtype).T

        self.n_bdr = len(xy_l) + len(xy_r) + len(xy_d) + len(xy_u)

        self.bc_data_dict = {
            "BC_u": np.vstack([xy_l, xy_r, xy_d, xy_u]),
            "BC_v": np.vstack([xy_l, xy_r, xy_d, xy_u]),
            "BC_right_p": xy_r,

            "OC_u": ob_xy.astype(dtype),
            "OC_v": ob_xy.astype(dtype),
            "OC_p": ob_xy.astype(dtype),
        }

        self.xy_dmn_test = np.vstack([np.random.rand(10) * (x_r - x_l) + x_l,
                                      np.random.rand(10) * (y_r - y_l) + y_l]).astype(dtype).T

    def execute_modify(self, data):
        data.replace_with_anchors(self.xy_dmn)

        if len(self.case.names["ICBCOCs"]) != 0:
            data.train_x_bc = np.vstack([self.bc_data_dict[key] for key in self.case.names["ICBCOCs"]])
        data.num_bcs = [len(self.bc_data_dict[key]) for key in self.case.names["ICBCOCs"]]

        # self.n_bc = sum(data.num_bcs)
        data.train_x = np.vstack([data.train_x_bc, data.train_x_all])
        data.test_x = np.vstack([data.train_x_bc, self.xy_dmn_test])

        return data

