import torch

# from .nn import NN
# from .. import activations
# from .. import initializers
# from ... import config
from deepxde.nn.pytorch.nn import NN
from deepxde.nn import activations
from deepxde.nn import initializers
from deepxde import config


class FNN(NN):
    """Fully-connected neural network. 
    Original deepxde doesn't support input_transform for pytorch. 
    This bug is fixed here, by setting input_transform as an input parameter."""

    # def __init__(self, layer_sizes, activation, kernel_initializer):
    def __init__(self, layer_sizes, activation, kernel_initializer, input_transform=None):
        super().__init__()
        if isinstance(activation, list):
            if not (len(layer_sizes) - 1) == len(activation):
                raise ValueError(
                    "Total number of activation functions do not match with sum of hidden layers and output layer!"
                )
            self.activation = list(map(activations.get, activation))
        else:
            self.activation = activations.get(activation)
        initializer = initializers.get(kernel_initializer)
        initializer_zero = initializers.get("zeros")
        
        self._input_transform = input_transform  # add
        n_input = layer_sizes[0]  # add

        if input_transform is not None:  # add
            test_x = torch.randn([1, n_input], dtype=config.real(torch))
            n_input_transform = input_transform(test_x).shape[1]
        else:
            n_input_transform = n_input

        self.linears = torch.nn.ModuleList()
        for i in range(1, len(layer_sizes)):  # modified
            # self.linears.append(
            #     torch.nn.Linear(
            #         layer_sizes[i - 1], layer_sizes[i], dtype=config.real(torch)
            #     )
            # )
            prev_layer_size = layer_sizes[i - 1] if i != 1 else n_input_transform
            curr_layer_size = layer_sizes[i]
            self.linears.append(torch.nn.Linear(
                prev_layer_size, curr_layer_size, dtype=config.real(torch)
            ))
            initializer(self.linears[-1].weight)
            initializer_zero(self.linears[-1].bias)

    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)
        for j, linear in enumerate(self.linears[:-1]):
            x = (
                self.activation[j](linear(x))
                if isinstance(self.activation, list)
                else self.activation(linear(x))
            )
        x = self.linears[-1](x)
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x




