from typing import Union

import torch
from torch import nn

Activation = Union[str, nn.Module]

_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}


class MlpPolicy(torch.nn.Module):
    def __init__(self, n_layers, hidden_size, input_size, output_size, hidden_act, output_act):
        super(MlpPolicy, self).__init__()
        self.in_layer = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.hidden_layers = [nn.Linear(in_features=hidden_size, out_features=hidden_size) for _ in range(n_layers)]
        self.hidden_layers = nn.ModuleList(self.hidden_layers)
        self.out_layer = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.hidden_activation = hidden_act
        self.out_activation = output_act

    def forward(self, x):
        x = self.in_layer(x)
        x = self.hidden_activation(x)
        for hlayer in self.hidden_layers:
            x = hlayer(x)
            x = self.hidden_activation(x)
        x = self.out_layer(x)
        x = self.out_activation(x)
        return x


def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int,
        activation: Activation = 'tanh',
        output_activation: Activation = 'identity',
) -> nn.Module:
    """
        Builds a feedforward neural network

        arguments:
            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer

            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer

        returns:
            MLP (nn.Module)
    """
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]
    mlp = MlpPolicy(n_layers, size, input_size, output_size, activation, output_activation)
    return mlp


device = None


def init_gpu(use_gpu=True, gpu_id=0):
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        print("Using GPU id {}".format(gpu_id))
    else:
        device = torch.device("cpu")
        print("GPU not detected. Defaulting to CPU.")


def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)


def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()
