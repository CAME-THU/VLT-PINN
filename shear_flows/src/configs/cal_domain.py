"""Define the calculation domain and its boundaries."""
import deepxde as dde
import numpy as np


class RectCalDomain:
    def __init__(self, xymin, xymax):
        self.xmin, self.ymin = xymin[0], xymin[1]
        self.xmax, self.ymax = xymax[0], xymax[1]
        self.geom = dde.geometry.Rectangle(xmin=xymin, xmax=xymax)

    def bdr_left(self, xy, on_boundary):
        x = xy[0]
        return on_boundary and np.isclose(x, self.xmin)

    # def bdr_right(self, xy, on_boundary):
    #     x = xy[0]
    #     return on_boundary and np.isclose(x, self.xmax)

    def bdr_down(self, xy, on_boundary):
        y = xy[1]
        return on_boundary and np.isclose(y, self.ymin)

    def bdr_up(self, xy, on_boundary):
        y = xy[1]
        return on_boundary and np.isclose(y, self.ymax)

    def bdr_left_right(self, xy, on_boundary):
        x = xy[0]
        return on_boundary and (np.isclose(x, self.xmin) or np.isclose(x, self.xmax))

    def bdr_down_up(self, xy, on_boundary):
        y = xy[1]
        return on_boundary and (np.isclose(y, self.ymin) or np.isclose(y, self.ymax))

    def bdr_left_right_up(self, xy, on_boundary):
        x, y = xy[0], xy[1]
        return on_boundary and (np.isclose(x, self.xmin) or np.isclose(x, self.xmax) or np.isclose(y, self.ymax))

