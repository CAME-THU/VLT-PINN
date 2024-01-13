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
delta = 4e-2
K = 1e-1  # K = J / rho, kinetic momentum flux
temp_flux = 1e0
alpha = 0.033
Pr_t = 0.84
beta = mt.gamma(Pr_t + 1) * mt.gamma(0.5) / mt.gamma(Pr_t + 1.5)

eta_half_umax = mt.atanh(2 ** 0.5 * 0.5)
width_half_umax = 2 * 4 * alpha * delta * eta_half_umax
umax = 0.25 * (3 * K / (alpha * delta)) ** (1 / 2)
Re_width = umax * width_half_umax / nu

save_dir = f"./results/jet_tur_plane/"
# save_dir += "Re{:.1f}_uin{:.2e}_nu{:.2e}/".format(Re_in, u_in_avg, nu)
# save_dir += "delta{:.2e}_K{:.2e}/".format(delta, K)
save_dir += "delta{:.2e}_K{:.2e}_ET{:.2e}/".format(delta, K, temp_flux)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# ----------------------------------------------------------------------
# define functions
def func_eta(x, y):
    return y / (4 * alpha * x)


def func_u(x, y):
    x_ = x + delta
    eta = func_eta(x_, y)
    tanh_eta = np.tanh(eta)
    u_max = 0.25 * (3 * K / (alpha * x_)) ** (1 / 2)
    return u_max * (1 - tanh_eta ** 2)


def func_v(x, y):
    x_ = x + delta
    eta = func_eta(x_, y)
    tanh_eta = np.tanh(eta)
    v_far = 0.5 * (3 * alpha * K / x_) ** (1 / 2)
    return v_far * (2 * eta * (1 - tanh_eta ** 2) - tanh_eta)


def func_T(x, y):
    x_ = x + delta
    eta = func_eta(x_, y)
    tanh_eta = np.tanh(eta)
    T_max = temp_flux / beta * (3 * alpha * K * x_) ** (-1 / 2)
    return T_max * (1 - tanh_eta ** 2) ** Pr_t


def func_psi(x, y):
    x_ = x + delta
    eta = func_eta(x_, y)
    tanh_eta = np.tanh(eta)
    return (3 * alpha * K * x_) ** (1 / 2) * tanh_eta


def func_nut(x, y):
    x_ = x + delta
    return (3 * alpha ** 3 * K * x_) ** (1 / 2)


# N_integral = 500
# inlet_ys = np.linspace(-b/2, b/2, N_integral)
# inlet_us = func_u(0, inlet_ys)
# inlet_u_avg_mass = np.sum(inlet_us * (b / N_integral)) / b
# inlet_u_avg_mom = (np.sum(inlet_us ** 2 * (b / N_integral)) / b) ** 0.5

# ----------------------------------------------------------------------
# plot the fields
x_l, x_r = 0.0, 1.0
y_l, y_r = -0.2, 0.2
n_x, n_y = 501, 801
x_lins = np.linspace(x_l, x_r, n_x)
y_lins = np.linspace(y_l, y_r, n_y)
xx, yy = np.meshgrid(x_lins, y_lins, indexing="ij")
# xy = np.vstack((np.ravel(xx), np.ravel(yy))).T

uu = func_u(xx, yy)
vv = func_v(xx, yy)
TT = func_T(xx, yy)
UU = np.sqrt(uu ** 2 + vv ** 2)
psipsi = func_psi(xx, yy)
nutnut = func_nut(xx, yy)


ffs = [uu, vv, TT, UU, psipsi, nutnut]
mathnames = ["$u$", "$v$", "$T$", "$U$", r"$\psi$", r"$\nu_t$"]
textnames = ["u", "v", "T", "U", "psi", "nut"]
units = ["m/s", "m/s", "K", "m/s", "m$^2$/s", "m$^2$/s"]

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

