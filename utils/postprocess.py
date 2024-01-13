import deepxde as dde
import numpy as np
# import torch
import os

import matplotlib
matplotlib.use("Agg")  # do not show figures
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
set_fs = 24
set_dpi = 200
plt.rcParams["font.sans-serif"] = "Times New Roman"  # default font
plt.rcParams["font.size"] = set_fs  # default font size
plt.rcParams["mathtext.fontset"] = "stix"  # default font of math text


class PostProcess:
    """Base class for post-processing.

    Args:
        args: args of the main function.
        case: case file variable.
        model: deepxde model.
        output_dir: output directory.
    """
    def __init__(self, args, case, model, output_dir):
        self.args = args
        self.case = case
        self.model = model
        self.output_dir = output_dir

        # The following 5 variables should be modified by inheritance, and they must have the same length.
        # Each item of preds and refes is ndarray, and their shape should match the problem dimension.
        # For example, (n_x, ) for 1D problems, (n_x, n_y) for 2D, (n_x, n_y, n_t) for 2Dt, etc.
        self.preds = []  # predicted field variables of interest
        self.refes = []  # corresponding reference field variables of interest
        self.mathnames = []  # the field variable names in math format, using for figure titles, for example
        self.textnames = []  # the field variable names in plain text format
        self.units = []  # the units of the field variables

    def save_data_(self, save_refe=True, suffix=""):
        """Save the predicted and reference fields."""
        print("Saving data...")
        output_dir = self.output_dir
        os.makedirs(output_dir + "data/", exist_ok=True)
        for i in range(len(self.preds)):
            np.save(output_dir + f"data/{self.textnames[i]}_{suffix}_predicted.npy", self.preds[i])
            if save_refe:
                np.save(output_dir + f"data/{self.textnames[i]}_{suffix}_reference.npy", self.refes[i])

    def save_metrics(self):
        """Save the evaluation metrics of predicted results w.r.t. reference results."""
        print("Saving metrics...")
        output_dir = self.output_dir
        refes = [field.ravel() for field in self.refes]
        preds = [field.ravel() for field in self.preds]

        formats = {"l2 relative error": "{:.4%}",
                   "MSE": "{:.2e}",
                   "MAPE": "{:.2e}",
                   "max APE": "{:.2e}",
                   }

        metrics = {}
        for key in formats.keys():
            metrics[key] = []
            metric_func = dde.metrics.get(key)
            for i in range(len(self.preds)):
                metric = metric_func(refes[i], preds[i])
                metrics[key].append(formats[key].format(metric))

        file = open(output_dir + "metrics.txt", "a")
        # file = open(output_dir + "metrics.txt", "w")
        file.write("\n")
        file.write("field variables:   ")
        file.write(", ".join(self.textnames))
        file.write("\n")
        for key in metrics.keys():
            file.write(key + ":   ")
            file.write(", ".join(metrics[key]))
            file.write("\n")
        file.write("\n")
        file.close()

    def save_var_metrics(self, vars_refe=(0.1, ), vars_infe=(dde.Variable(1.0),), vars_name=("nu",)):
        """Save the evaluation metrics of external trainable variables."""
        print("Saving metrics of external trainable variables...")
        output_dir = self.output_dir
        file = open(output_dir + "metrics.txt", "a")
        # file = open(output_dir + "metrics.txt", "w")
        file.write("\n")
        file.write("external trainable variables:   ")
        file.write(", ".join(vars_name))
        file.write("\n")

        file.write("reference:   ")
        for i in range(len(vars_refe)):
            file.write("{:.4e}".format(vars_refe[i]) + ", ")
        file.write("\n")

        file.write("inferred:   ")
        for i in range(len(vars_infe)):
            file.write("{:.4e}".format(vars_infe[i]) + ", ")
        file.write("\n")

        file.write("relative error:   ")
        for i in range(len(vars_infe)):
            file.write("{:.4%}".format(vars_infe[i] / vars_refe[i] - 1) + ", ")
        file.write("\n")
        file.close()

    def plot_save_loss_history(self):
        """Plot the loss history and save the history data."""
        print("Plotting and saving loss history...")
        model = self.model
        names = self.case.names
        output_dir = self.output_dir
        os.makedirs(output_dir + "pics/", exist_ok=True)

        loss_history = model.losshistory
        loss_train = np.array(loss_history.loss_train)
        loss_names = names["equations"] + names["ICBCOCs"]

        s_de = slice(0, len(names["equations"]), 1)
        s_bc = slice(len(names["equations"]), len(loss_names), 1)
        ss = [s_de, s_bc]

        # plot in two figures
        fig, axes = plt.subplots(1, 2, sharey="all", figsize=(15, 6))
        for i in range(2):
            axes[i].set_xlabel("Epoch")
            axes[i].set_yscale("log")
            axes[i].tick_params(axis="y", labelleft=True)
            axes[i].plot(loss_history.steps, loss_train[:, ss[i]])
            axes[i].legend(loss_names[ss[i]], fontsize="small")
        plt.savefig(output_dir + "pics/losses_train_2figs.png", bbox_inches="tight", dpi=set_dpi)
        plt.close(fig)

        # plot in one figure
        plt.figure(figsize=(8, 6))
        plt.xlabel("Epoch")
        plt.yscale("log")
        plt.plot(loss_history.steps, loss_train, lw=2)
        plt.legend(loss_names, fontsize="small")
        plt.savefig(output_dir + "pics/losses_train_1fig.png", bbox_inches="tight", dpi=set_dpi)
        plt.close(fig)

        # save the loss history
        loss_save = np.hstack([
            np.array(loss_history.steps)[:, None],
            np.array(loss_history.loss_train),
        ])
        np.savetxt(output_dir + "losses_train.txt", loss_save, fmt="%.2e", delimiter="    ",
                   header="    ".join(["step"] + loss_names))

    def plot_save_var_history(self, vars_refe=(0.1, ), scales=(1, ),
                              mathnames=(r"$\nu$", ), textnames=("nu", ), units=("m$^2$/s", )):
        """Plot the history of external trainable variables and save the history data."""
        print("Plotting and saving variable learning history...")
        output_dir = self.output_dir
        os.makedirs(output_dir + "data/", exist_ok=True)
        os.makedirs(output_dir + "pics/", exist_ok=True)

        file = open(self.output_dir + "vars_history_scaled.txt", "r")
        lines = file.readlines()
        epochs = np.array([int(line.split(" [")[0]) for line in lines])
        vars_history = np.array([line.split(" [")[1][:-2].split(", ") for line in lines], dtype=float)
        vars_history = vars_history / np.array(scales)
        file.close()

        for i in range(vars_history.shape[1]):
            plt.figure(figsize=(8, 6))
            plt.title(mathnames[i], fontsize="medium")
            plt.xlabel("Epoch")
            plt.ylabel(units[i])
            plt.plot(epochs, np.ones(len(epochs)) * vars_refe[i], c="k", ls="--", lw=3, label="Reference")
            plt.plot(epochs, vars_history[:, i], lw=2, label="PINN")
            plt.legend(fontsize="small")
            plt.savefig(output_dir + f"pics/variable{i + 1}_{textnames[i]}.png", bbox_inches="tight", dpi=set_dpi)
            plt.close()
        np.save(output_dir + "data/variables_history.npy", vars_history)

    def delete_old_models(self):
        """Delete the old models produced during training"""
        print("Deleting old models...")
        output_dir = self.output_dir
        model_list = os.listdir(output_dir + "models/")
        model_list_better = [s for s in model_list if "better" in s]
        better_epochs = [int(s.split("-")[1][:-3]) for s in model_list_better]
        best_epoch_index = better_epochs.index(max(better_epochs))
        for filename in model_list_better:
            if filename != model_list_better[best_epoch_index]:
                os.remove(output_dir + "models/" + filename)

    def plot_lossgrads_fnn(self):
        """Plot the distribution of the losses gradients w.r.t. the NN parameters. Only valid for FNNs."""
        print("Plotting loss gradients...")
        # TODO


class PostProcess1D(PostProcess):
    """Post-processing for 1D problems (1D space or 1D time)."""
    def __init__(self, args, case, model, output_dir):
        super().__init__(args, case, model, output_dir)
        self.n_x = 5001
        self.x = np.linspace(case.x_l, case.x_r, self.n_x)  # can be modified to t_l if the only dimension is t
        self.x_name = "$x$"  # can be modified by inheritance, to $t$" for example
        self.x_unit = "m"  # can be modified by inheritance, to "s" for example

    def save_data(self, save_refe=True):
        """Save the predicted and reference fields."""
        self.save_data_(save_refe=save_refe, suffix="1D")

    def plot_1dfields(self, lw=2, format="png"):
        """Plot the curves of predicted 1D fields."""
        print("Plotting 1D fields...")
        output_dir = self.output_dir
        os.makedirs(output_dir + "pics/", exist_ok=True)
        x = self.x
        preds = self.preds
        mnames, tnames, units = self.mathnames, self.textnames, self.units

        for i in range(len(preds)):
            plt.figure(figsize=(8, 6))
            plt.title(mnames[i], fontsize="medium")
            plt.xlabel(f"{self.x_name}/{self.x_unit}")
            plt.ylabel(units[i])
            plt.plot(x, preds[i], lw=lw)
            plt.savefig(output_dir + f"pics/field{i+1}_{tnames[i]}.{format}", bbox_inches="tight", dpi=set_dpi)
            plt.close()

    def plot_1dfields_comp(self, lws=(1, 2.5), label_refe="Reference", label_pred="PINN",
                           fsize_legend=set_fs-6, format="png"):
        """Plot the curves of predicted v.s. reference 1D fields."""
        print("Plotting 1D fields (predicted vs reference)...")
        output_dir = self.output_dir
        os.makedirs(output_dir + "pics/", exist_ok=True)
        x = self.x
        preds = self.preds
        refes = self.refes
        mnames, tnames, units = self.mathnames, self.textnames, self.units

        for i in range(len(preds)):
            plt.figure(figsize=(8, 6))
            plt.title(mnames[i], fontsize="medium")
            plt.xlabel(f"{self.x_name}/{self.x_unit}")
            plt.ylabel(units[i])
            plt.plot(x, refes[i], c="C0", ls="-", lw=lws[0], label=label_refe)
            plt.plot(x, preds[i], c="C1", ls="--", lw=lws[1], label=label_pred)
            plt.legend(fontsize=fsize_legend)
            plt.savefig(output_dir + f"pics/fieldComp{i+1}_{tnames[i]}.{format}", bbox_inches="tight", dpi=set_dpi)
            plt.close()


class PostProcess1Dt(PostProcess):
    """Post-processing for 1D unsteady problems, i.e. the independent variables are x and t."""
    def __init__(self, args, case, model, output_dir):
        super().__init__(args, case, model, output_dir)
        # TODO: Complete the code.


class PostProcess2D(PostProcess):
    """Post-processing for 2D steady problems, i.e. the independent variables are x and y."""
    def __init__(self, args, case, model, output_dir):
        super().__init__(args, case, model, output_dir)
        self.n_x, self.n_y = 501, 501
        self.x = np.linspace(case.x_l, case.x_r, self.n_x)
        self.y = np.linspace(case.y_l, case.y_r, self.n_y)
        self.xx, self.yy = np.meshgrid(self.x, self.y, indexing="ij")
        self.x_name, self.y_name = "$x$", "$y$"
        self.x_unit, self.y_unit = "m", "m"

    def save_data(self, save_refe=True):
        """Save the predicted and reference fields."""
        self.save_data_(save_refe=save_refe, suffix="2D")

    def plot_sampling_points(self, figsize=(8, 6), format="png"):
        """Plot the sampling points, including PDE points and IC/BC/OC points."""
        print("Plotting sampling points...")
        model = self.model
        output_dir = self.output_dir
        os.makedirs(output_dir + "pics/", exist_ok=True)
        x_l, x_r, y_l, y_r = self.case.x_l, self.case.x_r, self.case.y_l, self.case.y_r
        scale_x, scale_y = self.args.scale_x, self.args.scale_y
        shift_x, shift_y = self.args.shift_x, self.args.shift_y

        plt.figure(figsize=figsize)
        plt.axis("scaled")
        plt.axis([x_l - 0.05 * (x_r - x_l), x_r + 0.05 * (x_r - x_l),
                  y_l - 0.05 * (y_r - y_l), y_r + 0.05 * (y_r - y_l)])
        plt.xlabel(f"{self.x_name}/{self.x_unit}")
        plt.ylabel(f"{self.y_name}/{self.y_unit}")
        plt.scatter(model.data.train_x_all[:, 0] / scale_x - shift_x,
                    model.data.train_x_all[:, 1] / scale_y - shift_y, s=0.2, lw=0.2)
        plt.scatter(model.data.train_x_bc[:, 0] / scale_x - shift_x,
                    model.data.train_x_bc[:, 1] / scale_y - shift_y, s=5, marker="x", lw=0.5)
        # plt.scatter(data.anchors[:, 0], data.anchors[:, 1], s=0.5)
        plt.savefig(output_dir + f"pics/sampling_points.{format}", bbox_inches="tight", dpi=500)  # .svg
        plt.close()

    def plot_2dfields(self, figsize=(8, 6), format="png"):
        """Plot the contours of predicted 2D fields, and streamline for flow problems."""
        print("Plotting 2D fields...")
        output_dir = self.output_dir
        os.makedirs(output_dir + "pics/", exist_ok=True)
        x_l, x_r, y_l, y_r = self.case.x_l, self.case.x_r, self.case.y_l, self.case.y_r
        xx, yy = self.xx, self.yy
        preds = self.preds
        mnames, tnames, units = self.mathnames, self.textnames, self.units

        n_level = 50
        for i in range(len(preds)):
            plt.figure(figsize=figsize)
            plt.title(mnames[i], fontsize="medium")
            plt.axis("scaled")
            plt.axis([x_l, x_r, y_l, y_r])
            plt.xlabel(f"{self.x_name}/{self.x_unit}")
            plt.ylabel(f"{self.y_name}/{self.y_unit}")
            if tnames[i] == "psi":
                plt.contour(xx, yy, preds[i], colors="k", levels=n_level, linewidths=0.8, linestyles="solid")
            else:
                plt.contourf(xx, yy, preds[i], cmap="jet", levels=n_level)
                plt.colorbar(label=units[i], format="%.1e", pad=0.1)
            plt.savefig(output_dir + f"pics/contour{i+1}_{tnames[i]}.{format}", bbox_inches="tight", dpi=set_dpi)
            plt.close()

    def plot_2dfields_comp(self, figsize=(13, 6), is_vertical=False, label_refe="Reference", label_pred="Predicted",
                           adjust=(0.1, 0.8, None, None, None, None), format="png"):
        """Plot the contours of predicted v.s. reference 2D fields."""
        print("Plotting 2D fields (predicted vs reference)...")
        output_dir = self.output_dir
        os.makedirs(output_dir + "pics/", exist_ok=True)
        x_l, x_r, y_l, y_r = self.case.x_l, self.case.x_r, self.case.y_l, self.case.y_r
        xx, yy = self.xx, self.yy
        preds = self.preds
        refes = self.refes
        mnames, tnames, units = self.mathnames, self.textnames, self.units

        n_row, n_col = (2, 1) if is_vertical else (1, 2)
        n_level = 50
        for i in range(len(preds)):
            fig, axes = plt.subplots(n_row, n_col, figsize=figsize)
            f_min = min(preds[i].min(), refes[i].min())
            f_max = max(preds[i].max(), refes[i].max())
            norm = matplotlib.colors.Normalize(vmin=f_min, vmax=f_max)
            subfigs = []
            for j in range(2):
                axes[j].set_title(f"{[label_refe, label_pred][j]} {mnames[i]}", fontsize="medium")
                axes[j].axis("scaled")
                axes[j].axis([x_l, x_r, y_l, y_r])
                if tnames[i] == "psi":
                    subfig = axes[j].contour(xx, yy, [refes, preds][j][i],
                                             colors="k", levels=n_level, linewidths=0.8, linestyles="solid")
                else:
                    subfig = axes[j].contourf(xx, yy, [refes, preds][j][i], cmap="jet", levels=n_level, norm=norm)
                subfigs.append(subfig)
            if is_vertical:
                axes[0].set_xticklabels([])
                axes[1].set_xlabel(f"{self.x_name}/{self.x_unit}")
                axes[0].set_ylabel(f"{self.y_name}/{self.y_unit}")
                axes[1].set_ylabel(f"{self.y_name}/{self.y_unit}")
            else:
                axes[0].set_xlabel(f"{self.x_name}/{self.x_unit}")
                axes[1].set_xlabel(f"{self.x_name}/{self.x_unit}")
                axes[0].set_ylabel(f"{self.y_name}/{self.y_unit}")
                axes[1].set_yticklabels([])

            fig.subplots_adjust(left=adjust[0], right=adjust[1], bottom=adjust[2], top=adjust[3],
                                wspace=adjust[4], hspace=adjust[5])
            if tnames[i] != "psi":
                cb_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])  # l, b, w, h
                cb = fig.colorbar(subfigs[0], cax=cb_ax, label=units[i], format="%.1e")
                # cb.set_ticks(np.linspace(f_min, f_max, 8))
            plt.savefig(output_dir + f"pics/contourComp{i+1}_{tnames[i]}.{format}", bbox_inches="tight", dpi=set_dpi)
            plt.close(fig)

    def plot_1dcurves(self, select_x=(0., 1., ), select_y=(0., 1., ), lws=(1, 2.5),
                      label_refe="reference", label_pred="PINN",
                      n_col=2, fsize_legend=set_fs-6, save_data=False, format="png"):
        """Plot the 1D curves at some lines of predicted v.s. reference 2D fields."""
        print("Plotting 1D curves...")
        output_dir = self.output_dir
        os.makedirs(output_dir + "pics/", exist_ok=True)
        if save_data:
            os.makedirs(output_dir + "data/", exist_ok=True)
        x, y = self.x, self.y
        preds = self.preds
        refes = self.refes
        mnames, tnames, units = self.mathnames, self.textnames, self.units

        # field variables vs x at different y
        label_order = [2 * k for k in range(len(select_y))] + [2 * k + 1 for k in range(len(select_y))]
        for i in range(len(preds)):
            plt.figure(figsize=(8, 6))
            plt.title(mnames[i], fontsize="medium")
            plt.xlabel(f"{self.x_name}/{self.x_unit}")
            plt.ylabel(units[i])
            for j in range(len(select_y)):
                s = "{:.2f}".format(select_y[j])
                loc = np.sum(select_y[j] >= y) - 1
                loc = loc - 1 if loc == len(x) - 1 else loc
                d = (select_y[j] - y[loc]) / (y[loc + 1] - y[loc])
                f_refe = (1 - d) * refes[i][:, loc] + d * refes[i][:, loc + 1]
                f_pred = (1 - d) * preds[i][:, loc] + d * preds[i][:, loc + 1]
                plt.plot(x, f_refe, ls="-", c=f"C{j}", lw=lws[0],
                         label=f"{self.y_name} = {s}{self.y_unit}, {label_refe}")
                plt.plot(x, f_pred, ls="--", c=f"C{j}", lw=lws[1],
                         label=f"{self.y_name} = {s}{self.y_unit}, {label_pred}")
                if save_data:
                    np.save(output_dir + f"data/curve_{tnames[i]}-x_y{s}_reference.npy", f_refe)
                    np.save(output_dir + f"data/curve_{tnames[i]}-x_y{s}_predicted.npy", f_pred)
            handles, labels = plt.gca().get_legend_handles_labels()
            plt.legend([handles[idx] for idx in label_order], [labels[idx] for idx in label_order],
                       ncol=n_col, labelspacing=0.3, columnspacing=0.5, fontsize=fsize_legend)
            plt.savefig(output_dir + f"pics/curve{i + 1}_{tnames[i]}-x.{format}", bbox_inches="tight", dpi=set_dpi)
            plt.close()

        # field variables vs y at different x
        label_order = [2 * k for k in range(len(select_x))] + [2 * k + 1 for k in range(len(select_x))]
        for i in range(len(preds)):
            plt.figure(figsize=(8, 6))
            plt.title(mnames[i], fontsize="medium")
            plt.xlabel(f"{self.y_name}/{self.y_unit}")
            plt.ylabel(units[i])
            for j in range(len(select_x)):
                s = "{:.2f}".format(select_x[j])
                loc = np.sum(select_x[j] >= x) - 1
                loc = loc - 1 if loc == len(x) - 1 else loc
                d = (select_x[j] - x[loc]) / (x[loc + 1] - x[loc])
                f_refe = (1 - d) * refes[i][loc, :] + d * refes[i][loc + 1, :]
                f_pred = (1 - d) * preds[i][loc, :] + d * preds[i][loc + 1, :]
                plt.plot(y, f_refe, ls="-", c=f"C{j}", lw=lws[0],
                         label=f"{self.x_name} = {s}{self.x_unit}, {label_refe}")
                plt.plot(y, f_pred, ls="--", c=f"C{j}", lw=lws[1],
                         label=f"{self.x_name} = {s}{self.x_unit}, {label_pred}")
                if save_data:
                    np.save(output_dir + f"data/curve_{tnames[i]}-y_x{s}_reference.npy", f_refe)
                    np.save(output_dir + f"data/curve_{tnames[i]}-y_x{s}_predicted.npy", f_pred)
            handles, labels = plt.gca().get_legend_handles_labels()
            plt.legend([handles[idx] for idx in label_order], [labels[idx] for idx in label_order],
                       ncol=n_col, labelspacing=0.3, columnspacing=0.5, fontsize=fsize_legend)
            plt.savefig(output_dir + f"pics/curve{i + 1}_{tnames[i]}-y.{format}", bbox_inches="tight", dpi=set_dpi)
            plt.close()

        if save_data:
            np.save(output_dir + f"data/curve_x.npy", x)
            np.save(output_dir + f"data/curve_y.npy", y)

    @staticmethod
    def stream_function(uu, vv, dx, dy, psi0=0):
        psipsi = np.ones_like(uu) * psi0
        u_x0_mid = 0.5 * (uu[0, 1:] + uu[0, :-1])
        psipsi[0, 1:] = psi0 + np.cumsum(u_x0_mid * dy)
        v_mid = 0.5 * (vv[1:, :] + vv[:-1, :])
        psipsi[1:, :] = psipsi[0:1, :] + np.cumsum(-v_mid * dx, axis=0)
        return psipsi


class PostProcess2Dt(PostProcess):
    """Post-processing for 2D unsteady problems, i.e. the independent variables are x, y, and t."""
    def __init__(self, args, case, model, output_dir):
        super().__init__(args, case, model, output_dir)
        # TODO: Complete the code.

