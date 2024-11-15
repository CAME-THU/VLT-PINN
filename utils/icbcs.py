"""   """

import numbers
# import numpy as np

# from .. import backend as bkd
# from .. import config
# from .. import data
# from .. import gradients as grad
# from .. import utils
# from ..backend import backend_name

# import deepxde as dde
from deepxde import backend as bkd
# from deepxde import config
from deepxde import data
from deepxde import gradients as grad
# from deepxde import utils
# from deepxde.backend import backend_name

from deepxde.icbc import IC, DirichletBC, NeumannBC, RobinBC, PeriodicBC, PointSetBC, PointSetOperatorBC
from deepxde.icbc.boundary_conditions import backend_name


class ScaledIC(IC):
    """Initial conditions: y([x, t0]) = func([x, t0])."""
    # TODO


class ScaledDirichletBC(DirichletBC):
    """Dirichlet boundary conditions: y(x) = func(x)."""

    def __init__(self, geom, func, on_boundary, component=0, scale=1.0):
        super().__init__(geom, func, on_boundary, component)
        self.scale = scale

    def error(self, X, inputs, outputs, beg, end, aux_var=None):
        values = self.func(X, beg, end, aux_var)
        if bkd.ndim(values) == 2 and bkd.shape(values)[1] != 1:
            raise RuntimeError(
                "DirichletBC function should return an array of shape N by 1 for each "
                "component. Use argument 'component' for different output components."
            )
        return (outputs[beg:end, self.component: self.component + 1] - values) * self.scale


class ScaledNeumannBC(NeumannBC):
    """Neumann boundary conditions: dy/dn(x) = func(x)."""

    def __init__(self, geom, func, on_boundary, component=0, scale=1.0):
        super().__init__(geom, func, on_boundary, component)
        self.scale = scale

    def error(self, X, inputs, outputs, beg, end, aux_var=None):
        values = self.func(X, beg, end, aux_var)
        return (self.normal_derivative(X, inputs, outputs, beg, end) - values) * self.scale


class ScaledRobinBC(RobinBC):
    """Robin boundary conditions: dy/dn(x) = func(x, y)."""
    # TODO


class ScaledPeriodicBC(PeriodicBC):
    """Periodic boundary conditions on component_x."""
    # TODO


class ScaledPointSetBC(PointSetBC):
    """Dirichlet boundary condition for a set of points.

    Compare the output (that associates with `points`) with `values` (target data).
    If more than one component is provided via a list, the resulting loss will
    be the addative loss of the provided componets.

    Args:
        points: An array of points where the corresponding target values are known and
            used for training.
        values: A scalar or a 2D-array of values that gives the exact solution of the problem.
        component: Integer or a list of integers. The output components satisfying this BC.
            List of integers only supported for the backend PyTorch.
        batch_size: The number of points per minibatch, or `None` to return all points.
            This is only supported for the backend PyTorch and PaddlePaddle.
            Note, If you want to use batch size here, you should also set callback
            'dde.callbacks.PDEPointResampler(bc_points=True)' in training.
        shuffle: Randomize the order on each pass through the data when batching.
    """

    def __init__(self, points, values, component=0, batch_size=None, shuffle=True, scale=1.0):
        super().__init__(points, values, component, batch_size, shuffle)
        self.scale = scale

    def error(self, X, inputs, outputs, beg, end, aux_var=None):
        slice_batch = slice(None) if self.batch_size is None else self.batch_indices
        slice_component = slice(self.component, self.component + 1
                                ) if isinstance(self.component, numbers.Number) else self.component
        term_outputs = outputs[beg:end, slice_component] * self.scale
        term_values = self.values[slice_batch] * self.scale
        return term_outputs - term_values


class ScaledPointSetOperatorBC(PointSetOperatorBC):
    # TODO
