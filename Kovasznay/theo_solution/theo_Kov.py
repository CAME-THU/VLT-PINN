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


# a, b, m, nu = 1, 0.25, 0, 0.005
a, b, m, nu = 1e-17, 0.25, 0, 0.005
# a, b, m, nu = 1e-4, 0.25, 0, 0.02
Re = 1 / nu
ksi = Re / 2 + (-1) ** m * mt.sqrt(0.25 * Re ** 2 + 4 * mt.pi ** 2 * b ** 2)


# x_l, x_r = -0.2, 0.0
x_l, x_r = 0.0, 0.2
y_l, y_r = 0.0, 1.0


def func_u(x, y):
    return 1 - a * np.exp(ksi * x) * np.cos(2 * np.pi * b * y)


def func_v(x, y):
    return a * ksi / (2 * np.pi * b) * np.exp(ksi * x) * np.sin(2 * np.pi * b * y)


def func_p(x, y):
    # return a ** 2 / 2 * (1 - np.exp(2 * ksi * x))
    return a ** 2 / 2 * (1 - np.exp(2 * ksi * x)) * (y ** 0)


save_dir = f"./results/Re{Re}_a{a}_b{b}_m{m}_{x_l}_{x_r}_{y_l}_{y_r}/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# ----------------------------------------------------------------------
# post-process
Nx, Ny = 501, 501
x_lins = np.linspace(x_l, x_r, Nx)
y_lins = np.linspace(y_l, y_r, Ny)
xx, yy = np.meshgrid(x_lins, y_lins, indexing="ij")
# xy = np.vstack((np.ravel(xx), np.ravel(yy))).T

uu = func_u(xx, yy)
vv = func_v(xx, yy)
pp = func_p(xx, yy)
umum = (uu ** 2 + vv ** 2) ** 0.5

# ----------------------------------------------------------------------
# plot
ffs = [uu, vv, pp]
textnames = ["u", "v", "p"]
mathnames = ["$u$", "$v$", "$p$"]
labels = ["m/s", "m/s", "Pa"]

fmt = "png"
# fmt = "svg"

n_level = 50
for i in range(len(ffs)):
    # plt.figure(figsize=(8, 6))
    plt.figure(figsize=(6, 8))
    plt.title(f"{mathnames[i]}")
    plt.axis("scaled")
    plt.axis([x_l, x_r, y_l, y_r])
    plt.xlabel("$x$/m")
    plt.ylabel("$y$/m")
    plt.contourf(xx, yy, ffs[i], cmap="jet", levels=n_level)

    # plt.colorbar(label=labels[i])
    cb = plt.colorbar(label=labels[i], orientation="vertical", pad=0.1, format="%.1e")
    cb.ax.tick_params(labelsize=20)
    cb.set_label(labels[i], size=20)

    plt.savefig(save_dir + f"contour{i + 1}_{textnames[i]}.{fmt}", bbox_inches="tight", dpi=set_dpi)
    plt.close()

# streamline and velocity vector
plt.figure(figsize=(8, 6))
# plt.title("Streamline and Velocity Vector")
plt.title("Streamline")
plt.axis("scaled")
plt.axis([x_l, x_r, y_l, y_r])
plt.xlabel("$x$/m")
plt.ylabel("$y$/m")
plt.streamplot(xx.T, yy.T, uu.T, vv.T,
               density=3, linewidth=0.8, minlength=0.1, maxlength=10.0,
               cmap="jet", color=umum.T, arrowsize=1, arrowstyle="-")
# plt.quiver(xx[::2, ::2], yy[::2, ::2], uu[::2, ::2], vv[::2, ::2], umum[::2, ::2],
#            cmap="jet", angles="xy", width=0.0015, headwidth=5, headlength=5)
plt.colorbar(label="m/s")
# plt.savefig(save_dir + "streamline_velocity_vector.png", bbox_inches="tight", set_dpi=set_dpi)
plt.savefig(save_dir + f"streamline.{fmt}", bbox_inches="tight", dpi=set_dpi)
plt.close()

# ----------------------------------------------------------------------
# plot selected curves
select_x = [x_l + 0.0 * (x_r - x_l),
            x_l + 0.2 * (x_r - x_l),
            x_l + 0.5 * (x_r - x_l),
            x_l + 0.8 * (x_r - x_l),
            x_l + 1.0 * (x_r - x_l)]
select_y = [y_l + 0.0 * (y_r - y_l),
            y_l + 0.2 * (y_r - y_l),
            y_l + 0.5 * (y_r - y_l),
            y_l + 0.8 * (y_r - y_l),
            y_l + 1.0 * (y_r - y_l)]


# funcs = [func_u, func_v, func_T]
# dep_names = ["u", "v", "T"]
# units = ["(m/s)", "(m/s)", "K"]
funcs = [func_u, func_v, func_p]
dep_names = ["u", "v", "p"]
units = ["(m/s)", "(m/s)", "Pa"]

for i in range(len(funcs)):
    plt.figure(figsize=(8, 6))
    plt.xlabel("$x$/m")
    plt.ylabel(f"${dep_names[i]}$/{units[i]}")
    for y in select_y:
        plt.plot(x_lins, funcs[i](x_lins, y), label="$y$ = {:.2f}m".format(y))
    plt.legend(fontsize=set_fs_legend)
    plt.savefig(save_dir + f"curve{i+1}_{dep_names[i]}_vs_x.png", bbox_inches="tight", dpi=set_dpi)
    plt.close()

for i in range(len(funcs)):
    plt.figure(figsize=(8, 6))
    plt.xlabel("$y$/m")
    plt.ylabel(f"${dep_names[i]}$/{units[i]}")
    for x in select_x:
        plt.plot(y_lins, funcs[i](x, y_lins), label="$x$ = {:.2f}m".format(x))
    plt.legend(fontsize=set_fs_legend)
    plt.savefig(save_dir + f"curve{i+1}_{dep_names[i]}_vs_y.png", bbox_inches="tight", dpi=set_dpi)
    plt.close()
