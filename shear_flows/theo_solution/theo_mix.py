import numpy as np
import os
from scipy.interpolate import interp1d

import matplotlib
matplotlib.use("Agg")  # do not show figures
import matplotlib.pyplot as plt
set_fs = 24
set_fs_legend = set_fs - 6  # "small": -4, "x-small": -8
set_dpi = 200
plt.rcParams["font.sans-serif"] = "Times New Roman"  # default font
plt.rcParams["font.size"] = set_fs  # default font size
plt.rcParams["mathtext.fontset"] = "stix"  # default font of math text


# ----------------------------------------------------------------------
# define constants
data = np.loadtxt("./mixing_layer.csv", delimiter=",")
func_f = interp1d(data[:, 0], data[:, 1], kind="linear")
func_f1 = interp1d(data[:, 0], data[:, 2], kind="linear")
func_f2 = interp1d(data[:, 0], data[:, 3], kind="linear")

u_far, nu = 0.15, 1.5e-5

delta = 0.1  # 0.1, 1
Re_x = u_far * delta / nu

# save_dir = f"./results/mixing_layer/Re{int(Re_x)}_U{u_far}_delta{delta}_nu{nu}/"
save_dir = f"./results/mixing_layer/"
save_dir += "Re{:.1f}_U{:}_delta{:}_nu{:.2e}/".format(Re_x, u_far, delta, nu)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# ----------------------------------------------------------------------
# define functions
def func_u(x, y):
    x_ = x + delta
    eta = y * (u_far / (nu * x_)) ** 0.5
    return u_far * func_f1(eta)


def func_v(x, y):
    x_ = x + delta
    eta = y * (u_far / (nu * x_)) ** 0.5
    return 0.5 * (nu * u_far / x_) ** 0.5 * (eta * func_f1(eta) - func_f(eta))


def func_psi(x, y):
    x_ = x + delta
    eta = y * (u_far / (nu * x_)) ** 0.5
    return (nu * u_far * x_) ** 0.5 * func_f(eta)


# ----------------------------------------------------------------------
# plot the fields
x_l, x_r = 0.0, 1.0
# y_l, y_r = 0.0, 0.2
# n_x, n_y = 501, 401
y_l, y_r = -0.2, 0.2
n_x, n_y = 501, 801
x_lins = np.linspace(x_l, x_r, n_x)
y_lins = np.linspace(y_l, y_r, n_y)
xx, yy = np.meshgrid(x_lins, y_lins, indexing="ij")
# xy = np.vstack((np.ravel(xx), np.ravel(yy))).T

uu = func_u(xx, yy)
vv = func_v(xx, yy)
# TT = func_T(xx, yy)
UU = np.sqrt(uu ** 2 + vv ** 2)
psipsi = func_psi(xx, yy)


# ffs = [uu, vv, TT, UU, psipsi]
# textnames = ["u", "v", "T", "U", "psi"]
# units = ["m/s", "m/s", "K", "m/s", "m$^2$/s"]
ffs = [uu, vv, UU, psipsi]
mathnames = ["$u$", "$v$", "$U$", r"$\psi$"]
textnames = ["u", "v", "U", "psi"]
units = ["m/s", "m/s", "m/s", "m$^2$/s"]

n_level = 50
for i in range(len(ffs)):
    plt.figure(figsize=(12, 6))
    plt.title(mathnames[i], fontsize="medium")
    plt.axis("scaled")
    plt.axis([x_l, x_r, y_l, y_r])
    plt.xlabel("$x$/m")
    plt.ylabel("$y$/m")
    plt.contourf(xx, yy, ffs[i], cmap="jet", levels=n_level)
    plt.colorbar(label=units[i])
    if textnames[i] == "psi":
        plt.contour(xx, yy, ffs[i], colors="k", levels=n_level, linewidths=0.3)
    plt.savefig(save_dir + f"contour{i+1}_{textnames[i]}.png", bbox_inches="tight", dpi=set_dpi)
    plt.close()

# ----------------------------------------------------------------------
# plot selected curves
select_x = [x_l + 0.02 * (x_r - x_l),
            x_l + 0.5 * (x_r - x_l),
            x_l + 1.0 * (x_r - x_l)]
select_y = [0.00, 0.1 * y_r, 0.25 * y_r]


# funcs = [func_u, func_v, func_T]
# textnames2 = ["u", "v", "T"]
# units2 = ["(m/s)", "(m/s)", "K"]
funcs = [func_u, func_v]
textnames2 = ["u", "v"]
units2 = ["(m/s)", "(m/s)"]

for i in range(len(funcs)):
    plt.figure(figsize=(8, 6))
    plt.xlabel("$x$/m")
    plt.ylabel(f"${textnames2[i]}$/{units2[i]}")
    for y in select_y:
        plt.plot(x_lins, funcs[i](x_lins, y), label="$y$ = {:.2f}m".format(y))
    plt.legend(fontsize=set_fs_legend)
    plt.savefig(save_dir + f"curve{i+1}_{textnames2[i]}_vs_x.png", bbox_inches="tight", dpi=set_dpi)
    plt.close()

for i in range(len(funcs)):
    plt.figure(figsize=(8, 6))
    plt.xlabel("$y$/m")
    plt.ylabel(f"${textnames2[i]}$/{units2[i]}")
    for x in select_x:
        plt.plot(y_lins, funcs[i](x, y_lins), label="$x$ = {:.2f}m".format(x))
    plt.legend(fontsize=set_fs_legend)
    plt.savefig(save_dir + f"curve{i+1}_{textnames2[i]}_vs_y.png", bbox_inches="tight", dpi=set_dpi)
    plt.close()

# a thick line
for i in range(len(funcs)):
    plt.figure(figsize=(8, 6))
    plt.xlabel("$y$/m")
    plt.ylabel(f"${textnames2[i]}$/{units2[i]}")
    plt.plot(y_lins, funcs[i](0.02, y_lins), lw=5)
    plt.savefig(save_dir + f"curve{i+1}_{textnames2[i]}_vs_x_0.02.png", bbox_inches="tight", dpi=set_dpi)
    plt.close()
