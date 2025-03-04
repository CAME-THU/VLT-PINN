"""Define the field variables to be post-processed."""
import numpy as np
from utils.postprocess import PostProcess2D


class Postprocess(PostProcess2D):
    def __init__(self, args, case, model, output_dir):
        super().__init__(args, case, model, output_dir)

        dx, dy = self.x[1] - self.x[0], self.y[1] - self.y[0]
        xy = np.vstack((np.ravel(self.xx), np.ravel(self.yy))).T

        # ----------------------------------------------------------------------
        # Get the predicted and reference fields
        output = model.predict(xy)
        u_pred = output[:, 0]
        v_pred = output[:, 1]
        u_refe = case.func_u(xy).ravel()
        v_refe = case.func_v(xy).ravel()

        self.uu_pred = u_pred.reshape([self.n_x, self.n_y])
        self.vv_pred = v_pred.reshape([self.n_x, self.n_y])
        self.UU_pred = np.sqrt(self.uu_pred ** 2 + self.vv_pred ** 2)
        self.psipsi_pred = self.stream_function(self.uu_pred, self.vv_pred, dx, dy)

        self.uu_refe = u_refe.reshape([self.n_x, self.n_y])
        self.vv_refe = v_refe.reshape([self.n_x, self.n_y])
        self.UU_refe = np.sqrt(self.uu_refe ** 2 + self.vv_refe ** 2)
        self.psipsi_refe = self.stream_function(self.uu_refe, self.vv_refe, dx, dy)

        self.preds += [self.uu_pred, self.vv_pred, self.UU_pred, self.psipsi_pred]
        self.refes += [self.uu_refe, self.vv_refe, self.UU_refe, self.psipsi_refe]
        self.mathnames += ["$u$", "$v$", r"$|\mathbf{U}|$", r"$\psi$"]
        self.textnames += ["u", "v", "Um", "psi"]
        self.units += ["m/s", "m/s", "m/s", "m$^2$/s"]

        if case.names["dependents"] == ["u", "v", "T"]:
            T_pred = output[:, 2]
            T_refe = case.func_T(xy).ravel()
            self.TT_pred = T_pred.reshape([self.n_x, self.n_y])
            self.TT_refe = T_refe.reshape([self.n_x, self.n_y])
            self.preds.insert(2, self.TT_pred)
            self.refes.insert(2, self.TT_refe)
            self.mathnames.insert(2, "$T$")
            self.textnames.insert(2, "T")
            self.units.insert(2, "K")

        if "nu" in args.infer_paras:
            self.para_infes += [case.nu_infe_s / args.scales["nu"], ]
            self.para_refes += [case.nu, ]
            self.para_mathnames += [r"$\nu$", ]
            self.para_textnames += ["nu", ]
            self.para_units += ["m$^2$/s", ]
