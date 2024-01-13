"""Define the field variables to be post-processed."""
import numpy as np
from utils.postprocess import PostProcess2D


class PostProcessKov(PostProcess2D):
    def __init__(self, args, case, model, output_dir):
        super().__init__(args, case, model, output_dir)

        dx, dy = self.x[1] - self.x[0], self.y[1] - self.y[0]
        xy = np.vstack((np.ravel(self.xx), np.ravel(self.yy))).T

        xs = (self.x + args.shift_x) * args.scale_x
        ys = (self.y + args.shift_y) * args.scale_y
        xsxs, ysys = np.meshgrid(xs, ys, indexing="ij")
        xy_s = np.vstack((np.ravel(xsxs), np.ravel(ysys))).T

        # ----------------------------------------------------------------------
        # Get the predicted and reference fields
        output = model.predict(xy_s)
        # self.output_terms = model.predict(xy_s, operator=case.terms_uv)
        u_pred = output[:, 0] / args.scale_u
        v_pred = output[:, 1] / args.scale_v
        p_pred = output[:, 2] / args.scale_p
        u_refe = case.func_u(xy).ravel()
        v_refe = case.func_v(xy).ravel()
        p_refe = case.func_p(xy).ravel()

        self.uu_pred = u_pred.reshape([self.n_x, self.n_y])
        self.vv_pred = v_pred.reshape([self.n_x, self.n_y])
        self.pp_pred = p_pred.reshape([self.n_x, self.n_y])
        self.UU_pred = np.sqrt(self.uu_pred ** 2 + self.vv_pred ** 2)
        self.psipsi_pred = self.stream_function(self.uu_pred, self.vv_pred, dx, dy)

        self.uu_refe = u_refe.reshape([self.n_x, self.n_y])
        self.vv_refe = v_refe.reshape([self.n_x, self.n_y])
        self.pp_refe = p_refe.reshape([self.n_x, self.n_y])
        self.UU_refe = np.sqrt(self.uu_refe ** 2 + self.vv_refe ** 2)
        self.psipsi_refe = self.stream_function(self.uu_refe, self.vv_refe, dx, dy)

        self.preds = [self.uu_pred, self.vv_pred, self.pp_pred, self.UU_pred, self.psipsi_pred]
        self.refes = [self.uu_refe, self.vv_refe, self.pp_refe, self.UU_refe, self.psipsi_refe]
        self.mathnames = ["$u$", "$v$", "$p$", r"$|\mathbf{U}|$", r"$\psi$"]
        self.textnames = ["u", "v", "p", "U", "psi"]
        self.units = ["m/s", "m/s", "Pa", "m/s", "m$^2$/s"]

