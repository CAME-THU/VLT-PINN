import numpy as np
import math as mt
import os

import matplotlib
matplotlib.use("Agg")  # do not show figures
import matplotlib.pyplot as plt
set_fs = 24  # + 8 + 8
set_fs_legend = set_fs - 6  # "small": -4, "x-small": -8
set_dpi = 200
plt.rcParams["font.sans-serif"] = "Times New Roman"  # default font
plt.rcParams["font.size"] = set_fs  # default font size
plt.rcParams["mathtext.fontset"] = "stix"  # default font of math text


# ----------------------------------------------------------------------
# define constants
nu = 1.5e-5  # nu = 1.7894e-5 / 1.225
delta = 5e-3
K = 1e-5  # K = J / rho, kinetic momentum flux
temp_flux = 1e-2
Pr = 0.71
beta = mt.gamma(Pr + 1) * mt.gamma(0.5) / mt.gamma(Pr + 1.5)

eta_half_umax = mt.atanh(2 ** 0.5 * 0.5)
width_half_umax = 2 * 3 * (4 / 3) ** (2 / 3) * (nu ** 2 / K) ** (1 / 3) * delta ** (2 / 3) * eta_half_umax
umax = (2 / 3) * (3 / 4) ** (4 / 3) * (K ** 2 / (nu * delta)) ** (1 / 3)
Re_width = umax * width_half_umax / nu

save_dir = f"./results/jet_lam_plane/"
# save_dir += "Re{:.1f}_uin{:.2e}_nu{:.2e}/".format(Re_in, u_in_avg, nu)
save_dir += "delta{:.2e}_K{:.2e}_ET{:.2e}/".format(delta, K, temp_flux)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# ----------------------------------------------------------------------
# define functions
def func_eta(x, y):
    c = (1 / 3) * (3 / 4) ** (2 / 3)
    return c * (K / nu ** 2) ** (1 / 3) * y * x ** (-2 / 3)


def func_u(x, y):
    x_ = x + delta
    eta = func_eta(x_, y)
    tanh_eta = np.tanh(eta)
    c = (2 / 3) * (3 / 4) ** (4 / 3)
    u_max = c * (K ** 2 / (nu * x_)) ** (1 / 3)
    return u_max * (1 - tanh_eta ** 2)


def func_v(x, y):
    x_ = x + delta
    eta = func_eta(x_, y)
    tanh_eta = np.tanh(eta)
    c = (2 / 3) * (3 / 4) ** (2 / 3)
    v_far = c * (K * nu / x_ ** 2) ** (1 / 3)
    return v_far * (2 * eta * (1 - tanh_eta ** 2) - tanh_eta)


def func_T(x, y):
    x_ = x + delta
    eta = func_eta(x_, y)
    tanh_eta = np.tanh(eta)
    c = (0.5 / beta) * (4 / 3) ** (2 / 3)
    T_max = c * temp_flux * (K * nu * x_) ** (-1 / 3)
    return T_max * (1 - tanh_eta ** 2) ** Pr


def func_psi(x, y):
    x_ = x + delta
    eta = func_eta(x_, y)
    tanh_eta = np.tanh(eta)
    c = 2 * (3 / 4) ** (2 / 3)
    return c * (K * nu * x_) ** (1 / 3) * tanh_eta


# N_integral = 500
# inlet_ys = np.linspace(-b/2, b/2, N_integral)
# inlet_us = func_u(0, inlet_ys)
# inlet_u_avg_mass = np.sum(inlet_us * (b / N_integral)) / b
# inlet_u_avg_mom = (np.sum(inlet_us ** 2 * (b / N_integral)) / b) ** 0.5

# ----------------------------------------------------------------------
# plot the fields
x_l, x_r = 0.0, 1.0
y_l, y_r = 0.0, 0.2
# y_l, y_r = -0.2, 0.2
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


ffs = [uu, vv, TT, UU, psipsi]
textnames = ["u", "v", "T", "U", "psi"]
mathnames = ["$u$", "$v$", "$T$", "$U$", r"$\psi$"]
units = ["m/s", "m/s", "K", "m/s", "m$^2$/s"]

fmt = "png"
# fmt = "svg"

n_level = 50
for i in range(len(ffs)):
    # plt.figure(figsize=(12, 6))
    plt.figure(figsize=(8, 6))  # horizontal
    # plt.figure(figsize=(8, 3))
    plt.title(mathnames[i], fontsize="medium")
    plt.axis("scaled")
    plt.axis([x_l, x_r, y_l, y_r])
    plt.xlabel("$x$/m")
    plt.ylabel("$y$/m")

    plt.contourf(xx, yy, ffs[i], cmap="jet", levels=n_level)

    # plt.colorbar(label=units[i])  # vertical
    cb = plt.colorbar(label=units[i], orientation="horizontal", pad=0.2)  # horizontal
    # cb = plt.colorbar(label=units[i], orientation="vertical", pad=0.1)
    f1, f2 = np.min(ffs[i]), np.max(ffs[i])
    tks = [f1, 0.9 * f1 + 0.1 * f2,
           0.8 * f1 + 0.2 * f2, 0.7 * f1 + 0.3 * f2, 0.6 * f1 + 0.4 * f2, 0.5 * f1 + 0.5 * f2,
           0.4 * f1 + 0.6 * f2, 0.3 * f1 + 0.7 * f2, 0.2 * f1 + 0.8 * f2, 0.1 * f1 + 0.9 * f2, f2, ]
    # tks = [f1, 0.8 * f1 + 0.2 * f2, 0.6 * f1 + 0.4 * f2,
    #        0.4 * f1 + 0.6 * f2, 0.2 * f1 + 0.8 * f2, f2, ]
    cb.set_ticks(tks)
    cb.ax.set_xticklabels(["{:.1e}".format(tks[j]) for j in range(len(tks))], rotation=-90, size=set_fs_legend + 4)  # horizontal
    # cb.ax.set_yticklabels(["{:.1e}".format(tks[j]) for j in range(len(tks))], size=set_fs_legend + 4)  # vertical
    # cb.ax.tick_params(labelsize=20)
    cb.set_label(units[i], size=20)

    if textnames[i] == "psi":
        plt.contour(xx, yy, ffs[i], colors="k", levels=n_level, linewidths=0.3)
    plt.savefig(save_dir + f"contour{i+1}_{textnames[i]}.{fmt}", bbox_inches="tight", dpi=set_dpi)
    # plt.savefig(save_dir + f"contour{i+1}_{names_field[i]}.{fmt}", dpi=set_dpi)
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

# a thick line
for i in range(len(funcs)):
    plt.figure(figsize=(8, 6))
    plt.xlabel("$y$/m")
    plt.ylabel(f"${textnames2[i]}$/{units2[i]}")
    plt.plot(y_lins, funcs[i](0.02, y_lins), lw=5)
    plt.savefig(save_dir + f"curve{i+1}_{textnames2[i]}_vs_y_x0.02.png", bbox_inches="tight", dpi=set_dpi)
    plt.close()

for i in range(len(funcs)):
    plt.figure(figsize=(8, 6))
    plt.xlabel("$x$/m")
    plt.ylabel(f"${textnames2[i]}$/{units2[i]}")
    plt.plot(x_lins, funcs[i](x_lins, 0.02), lw=5)
    plt.savefig(save_dir + f"curve{i+1}_{textnames2[i]}_vs_x_y0.02.png", bbox_inches="tight", dpi=set_dpi)
    plt.close()
