import numpy as np
import math as mt
import os

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
nu = 1.5e-5  # nu = 1.7894e-5 / 1.225
delta = 6e-2
K = 1e-3  # K = J / rho, kinetic momentum flux
temp_flux = 1e-2
alpha = 0.017
nut = alpha * (3 * K / mt.pi) ** 0.5
Pr_t = 0.84

eta_half_umax = (2 ** 0.5 - 1) ** 0.5
width_half_umax = 2 * 8 * alpha * delta * eta_half_umax
umax = (0.125 / alpha) * (3 * K / mt.pi) ** 0.5 * delta ** (-1)
Re_width = umax * width_half_umax / nu

save_dir = f"./results/jet_tur_round/"
# save_dir += "Re{:.1f}_uin{:.2e}_nu{:.2e}/".format(Re_in, u_in_avg, nu)
# save_dir += "delta{:.2e}_K{:.2e}/".format(delta, K)
save_dir += "delta{:.2e}_K{:.2e}_ET{:.2e}/".format(delta, K, temp_flux)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# ----------------------------------------------------------------------
# define functions
def func_eta(x, y):
    return y / (8 * alpha * x)


def func_u(x, y):
    x_ = x + delta
    eta = func_eta(x_, y)
    # u_max = 3 * K / (8 * mt.pi * nut * x_)
    u_max = (0.125 / alpha) * (3 * K / mt.pi) ** 0.5 * x_ ** (-1)  # same as last line
    return u_max * (1 + eta ** 2) ** (-2)


def func_v(x, y):
    x_ = x + delta
    eta = func_eta(x_, y)
    v_max_x4 = 0.5 * (3 * K / mt.pi) ** 0.5 * x_ ** (-1)
    return v_max_x4 * (eta - eta ** 3) * (1 + eta ** 2) ** (-2)


def func_T(x, y):
    x_ = x + delta
    eta = func_eta(x_, y)
    T_max = (1 + 2 * Pr_t) * temp_flux / (8 * alpha * (3 * mt.pi * K) ** 0.5 * x_)
    return T_max * (1 + eta ** 2) ** (-2 * Pr_t)


def func_psi(x, y):
    x_ = x + delta
    eta = func_eta(x_, y)
    return nut * x_ * 4 * eta / (1 + eta ** 2)


# N_integral = 500
# inlet_rs = np.linspace(0.0, 0.5*b, N_integral)
# inlet_us = func_u(0, inlet_rs)
# inlet_u_avg_mass = np.sum(inlet_us * inlet_rs * (0.5*b / N_integral)) * (8 / b**2)
# inlet_u_avg_mom = (np.sum(inlet_us**2 * inlet_rs * (b / N_integral)) * (8 / b**2)) ** 0.5

# ----------------------------------------------------------------------
# plot the fields
x_l, x_r = 0.0, 1.0
y_l, y_r = -0.2, 0.2
# x_l, x_r = 0.0, 1.0 / 10
# y_l, y_r = -0.2 / 10, 0.2 / 10
n_x, n_r = 501, 801
x_lins = np.linspace(x_l, x_r, n_x)
y_lins = np.linspace(y_l, y_r, n_r)
xx, yy = np.meshgrid(x_lins, y_lins, indexing="ij")
# xy = np.vstack((np.ravel(xx), np.ravel(yy))).T

uu = func_u(xx, yy)
vv = func_v(xx, yy)
TT = func_T(xx, yy)
UU = np.sqrt(uu ** 2 + vv ** 2)
psipsi = func_psi(xx, yy)


ffs = [uu, vv, TT, UU, psipsi]
mathnames = ["$u$", "$v$", "$T$", "$U$", r"$\psi$"]
textnames = ["u", "v", "T", "U", "psi"]
units = ["m/s", "m/s", "K", "m/s", "m$^3$/s"]

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


funcs = [func_u, func_v, func_T]
textnames2 = ["u", "v", "T"]
units2 = ["(m/s)", "(m/s)", "K"]

for i in range(len(funcs)):
    plt.figure(figsize=(8, 6))
    plt.xlabel("$x$/m")
    plt.ylabel(f"${textnames2[i]}$/{units2[i]}")
    for y in select_y:
        plt.plot(x_lins, funcs[i](x_lins, y), label="$y$ = {:.3f}m".format(y))
    plt.legend(fontsize=set_fs_legend)
    plt.savefig(save_dir + f"curve{i+1}_{textnames2[i]}_vs_x.png", bbox_inches="tight", dpi=set_dpi)
    plt.close()

for i in range(len(funcs)):
    plt.figure(figsize=(8, 6))
    plt.xlabel("$y$/m")
    plt.ylabel(f"${textnames2[i]}$/{units2[i]}")
    for x in select_x:
        plt.plot(y_lins, funcs[i](x, y_lins), label="$x$ = {:.3f}m".format(x))
    plt.legend(fontsize=set_fs_legend)
    plt.savefig(save_dir + f"curve{i+1}_{textnames2[i]}_vs_y.png", bbox_inches="tight", dpi=set_dpi)
    plt.close()
