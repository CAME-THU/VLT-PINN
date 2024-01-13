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

        spc_base = (x_r - x_l) / 200

        exp_base = 10
        n_dmn_x, n_dmn_y = 300, 60
        n_bdr_x, n_bdr_y = int((x_r - x_l) / spc_base) + 1, int((y_r - y_l) / spc_base) + 1

        if args.case_name[0:3] == "jet":
            xs_dmn = my_logspace(xs_l, xs_r, n_dmn_x, exp_base, "start")
            xs_bdr = my_logspace(xs_l, xs_r, n_bdr_x, exp_base, "start")
        else:
            xs_dmn = np.linspace(xs_l, xs_r, n_dmn_x)
            xs_bdr = np.linspace(xs_l, xs_r, n_bdr_x)

        if args.case_name[0:3] == "mix":
            ys_dmn = my_logspace(ys_l, ys_r, n_dmn_y, exp_base, "middle")
            ys_bdr = my_logspace(ys_l, ys_r, n_bdr_y, exp_base, "middle")
        else:
            ys_dmn = my_logspace(ys_l, ys_r, n_dmn_y, exp_base, "start")
            ys_bdr = my_logspace(ys_l, ys_r, n_bdr_y, exp_base, "start")

        # if args.case_name == "boundary_layer":
        #     ys_dmn = my_logspace(ys_l, ys_r, n_dmn_y, exp_base, "start")
        #     ys_bdr = my_logspace(ys_l, ys_r, n_bdr_y, exp_base, "start")
        # else:
        #     ys_dmn = my_logspace(ys_l, ys_r, n_dmn_y, exp_base, "middle")
        #     ys_bdr = my_logspace(ys_l, ys_r, n_bdr_y, exp_base, "middle")

        # modify sampling points in domain
        xsxs_dmn, ysys_dmn = np.meshgrid(xs_dmn, ys_dmn, indexing="ij")
        self.xy_s_dmn = np.vstack((np.ravel(xsxs_dmn), np.ravel(ysys_dmn))).T
        self.n_dmn = n_dmn_x * n_dmn_y

        # modify sampling points on boundary
        xy_s_l = np.vstack([np.ones_like(ys_bdr) * xs_l, ys_bdr]).astype(dtype).T
        xy_s_r = np.vstack([np.ones_like(ys_bdr) * xs_r, ys_bdr]).astype(dtype).T
        xy_s_d = np.vstack([xs_bdr, np.ones_like(xs_bdr) * ys_l]).astype(dtype).T
        xy_s_u = np.vstack([xs_bdr, np.ones_like(xs_bdr) * ys_r]).astype(dtype).T

        # l_dense = (ys_r - ys_l) / 40
        # n_dense = int((n_bdr_y - 1) / 40 * 20) + 1
        # if case_name[0:3] == "mix":
        #     y_m = (ys_l + ys_r) / 2
        #     xy_leftdense = np.vstack([np.ones(n_dense) * xs_l,
        #                               np.linspace(y_m - l_dense / 2, y_m + l_dense / 2, n_dense)]).astype(dtype).T
        # else:
        #     xy_leftdense = np.vstack([np.ones(n_dense) * xs_l,
        #                               np.linspace(ys_l, ys_l + l_dense, n_dense)]).astype(dtype).T
        # xy_s_l = np.vstack([xy_s_l, xy_leftdense])

        self.n_bdr = len(xy_s_l) + len(xy_s_r) + len(xy_s_d) + len(xy_s_u)

        self.bc_data_dict = {
            "BC_left_u": xy_s_l,
            "BC_left_v": xy_s_l,
            "BC_sym_v": xy_s_d,
            "BC_sym_dudy": xy_s_d,
            "BC_far_u": xy_s_u,
            "BC_far_v": xy_s_u,
            "BC_all_u": np.vstack([xy_s_l, xy_s_r, xy_s_d, xy_s_u]),
            "BC_all_v": np.vstack([xy_s_l, xy_s_r, xy_s_d, xy_s_u]),
            "BC_left_right_u": np.vstack([xy_s_l, xy_s_r]),
            "BC_left_right_v": np.vstack([xy_s_l, xy_s_r]),
            "BC_down_up_u": np.vstack([xy_s_d, xy_s_u]),
            "BC_down_up_v": np.vstack([xy_s_d, xy_s_u]),
            "BC_left_right_up_u": np.vstack([xy_s_l, xy_s_r, xy_s_u]),

            "BC_left_T": xy_s_l,
            "BC_sym_dTdy": xy_s_d,
            "BC_left_right_up_T": np.vstack([xy_s_l, xy_s_r, xy_s_u]),

            "OC_u": ob_xy_s.astype(dtype),
            "OC_v": ob_xy_s.astype(dtype),
            "OC_T": ob_xy_s.astype(dtype),
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

