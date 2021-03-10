import torch.nn as nn
import numpy as np

import helpers


class Backbone(nn.Module):
    def __init__(self):
        """
        Backbone base class
        """
        super().__init__()
        self.layers = nn.ModuleList()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class CNN(Backbone):
    def __init__(self, cfg, flatten_output=True, **_):
        """
        CNN backbone

        :param cfg: CNN config
        :type cfg: config.defaults.CNN
        :param flatten_output: Flatten the backbone output?
        :type flatten_output: bool
        :param _:
        :type _:
        """
        super().__init__()

        self.output_size = list(cfg.input_size)

        for layer_type, *layer_params in cfg.layers:
            if layer_type == "conv":
                self.layers.append(nn.Conv2d(in_channels=self.output_size[0], out_channels=layer_params[2],
                                             kernel_size=layer_params[:2]))
                # Update output size
                self.output_size[0] = layer_params[2]
                self.output_size[1:] = helpers.conv2d_output_shape(self.output_size[1:], kernel_size=layer_params[:2])
                # Add activation
                if layer_params[3] == "relu":
                    self.layers.append(nn.ReLU())

            elif layer_type == "pool":
                self.layers.append(nn.MaxPool2d(kernel_size=layer_params))
                # Update output size
                self.output_size[1:] = helpers.conv2d_output_shape(self.output_size[1:], kernel_size=layer_params,
                                                                   stride=layer_params)

            elif layer_type == "relu":
                self.layers.append(nn.ReLU())

            elif layer_type == "lrelu":
                self.layers.append(nn.LeakyReLU(layer_params[0]))

            elif layer_type == "bn":
                if len(self.output_size) > 1:
                    self.layers.append(nn.BatchNorm2d(num_features=self.output_size[0]))
                else:
                    self.layers.append(nn.BatchNorm1d(num_features=self.output_size[0]))

            elif layer_type == "fc":
                self.layers.append(nn.Flatten())
                self.output_size = [np.prod(self.output_size)]
                self.layers.append(nn.Linear(self.output_size[0], layer_params[0], bias=True))
                self.output_size = [layer_params[0]]

            else:
                raise RuntimeError(f"Unknown layer type: {layer_type}")

        if flatten_output:
            self.layers.append(nn.Flatten())
            self.output_size = [np.prod(self.output_size)]


class MLP(Backbone):
    def __init__(self, cfg, input_size=None, **_):
        """
        MLP backbone

        :param cfg: MLP config
        :type cfg: config.defaults.MLP
        :param input_size: Optional input size which overrides the one set in `cfg`.
        :type input_size: Optional[Union[List, Tuple]]
        :param _:
        :type _:
        """
        super().__init__()
        self.output_size = self.create_linear_layers(cfg, self.layers, input_size=input_size)

    @staticmethod
    def get_activation_module(a):
        if a == "relu":
            return nn.ReLU()
        elif a == "sigmoid":
            return nn.Sigmoid()
        elif a == "tanh":
            return nn.Tanh()
        elif a == "softmax":
            return nn.Softmax(dim=1)
        elif a.startswith("leaky_relu"):
            neg_slope = float(a.split(":")[1])
            return nn.LeakyReLU(neg_slope)
        else:
            raise RuntimeError(f"Invalid MLP activation: {a}.")

    @classmethod
    def create_linear_layers(cls, cfg, layer_container, input_size=None):
        # `input_size` takes priority over `cfg.input_size`
        if input_size is not None:
            output_size = list(input_size)
        else:
            output_size = list(cfg.input_size)

        if len(output_size) > 1:
            layer_container.append(nn.Flatten())
            output_size = [np.prod(output_size)]

        n_layers = len(cfg.layers)
        activations = helpers.ensure_iterable(cfg.activation, expected_length=n_layers)
        use_bias = helpers.ensure_iterable(cfg.use_bias, expected_length=n_layers)
        use_bn = helpers.ensure_iterable(cfg.use_bn, expected_length=n_layers)

        for n_units, act, _use_bias, _use_bn in zip(cfg.layers, activations, use_bias, use_bn):
            # If we get n_units = -1, then the number of units should be the same as the previous number of units, or
            # the input dim.
            if n_units == -1:
                n_units = output_size[0]

            layer_container.append(nn.Linear(in_features=output_size[0], out_features=n_units, bias=_use_bias))
            if _use_bn:
                # Add BN before activation
                layer_container.append(nn.BatchNorm1d(num_features=n_units))
            if act is not None:
                # Add activation
                layer_container.append(cls.get_activation_module(act))
            output_size[0] = n_units

        return output_size


class Backbones(nn.Module):
    BACKBONE_CONSTRUCTORS = {
        "CNN": CNN,
        "MLP": MLP
    }

    def __init__(self, backbone_configs, flatten_output=True):
        """
        Class representing multiple backbones. Call with list of inputs, where inputs[0] goes into the first backbone,
        and so on.

        :param backbone_configs: List of backbone configs. Each element corresponds to a backbone.
        :type backbone_configs: List[Union[config.defaults.MLP, config.defaults.CNN], ...]
        :param flatten_output: Flatten the backbone outputs?
        :type flatten_output: bool
        """
        super().__init__()

        self.backbones = nn.ModuleList()
        for cfg in backbone_configs:
            self.backbones.append(self.create_backbone(cfg, flatten_output=flatten_output))

    @property
    def output_sizes(self):
        return [bb.output_size for bb in self.backbones]

    @classmethod
    def create_backbone(cls, cfg, flatten_output=True):
        if cfg.class_name not in cls.BACKBONE_CONSTRUCTORS:
            raise RuntimeError(f"Invalid backbone: '{cfg.class_name}'")
        return cls.BACKBONE_CONSTRUCTORS[cfg.class_name](cfg, flatten_output=flatten_output)

    def forward(self, views):
        assert len(views) == len(self.backbones), f"n_views ({len(views)}) != n_backbones ({len(self.backbones)})."
        outputs = [bb(v) for bb, v in zip(self.backbones, views)]
        return outputs


