import numpy as np
import torch as th
import torch.nn as nn


class _Fusion(nn.Module):
    def __init__(self, cfg, input_sizes):
        """
        Base class for the fusion module

        :param cfg: Fusion config. See config.defaults.Fusion
        :param input_sizes: Input shapes
        """
        super().__init__()
        self.cfg = cfg
        self.input_sizes = input_sizes
        self.output_size = None

    def forward(self, inputs):
        raise NotImplementedError()

    @classmethod
    def get_weighted_sum_output_size(cls, input_sizes):
        flat_sizes = [np.prod(s) for s in input_sizes]
        assert all(s == flat_sizes[0] for s in flat_sizes), f"Fusion method {cls.__name__} requires the flat output" \
                                                            f" shape from all backbones to be identical." \
                                                            f" Got sizes: {input_sizes} -> {flat_sizes}."
        return [flat_sizes[0]]

    def get_weights(self, softmax=True):
        out = []
        if hasattr(self, "weights"):
            out = self.weights
            if softmax:
                out = nn.functional.softmax(self.weights, dim=-1)
        return out

    def update_weights(self, inputs, a):
        pass


class Mean(_Fusion):
    def __init__(self, cfg, input_sizes):
        """
        Mean fusion.

        :param cfg: Fusion config. See config.defaults.Fusion
        :param input_sizes: Input shapes
        """
        super().__init__(cfg, input_sizes)
        self.output_size = self.get_weighted_sum_output_size(input_sizes)

    def forward(self, inputs):
        return th.mean(th.stack(inputs, -1), dim=-1)


class WeightedMean(_Fusion):
    """
    Weighted mean fusion.

    :param cfg: Fusion config. See config.defaults.Fusion
    :param input_sizes: Input shapes
    """
    def __init__(self, cfg, input_sizes):
        super().__init__(cfg, input_sizes)
        self.weights = nn.Parameter(th.full((self.cfg.n_views,), 1 / self.cfg.n_views), requires_grad=True)
        self.output_size = self.get_weighted_sum_output_size(input_sizes)

    def forward(self, inputs):
        return _weighted_sum(inputs, self.weights, normalize_weights=True)


def _weighted_sum(tensors, weights, normalize_weights=True):
    if normalize_weights:
        weights = nn.functional.softmax(weights, dim=0)
    out = th.sum(weights[None, None, :] * th.stack(tensors, dim=-1), dim=-1)
    return out


MODULES = {
    "mean": Mean,
    "weighted_mean": WeightedMean,
}


def get_fusion_module(cfg, input_sizes):
    return MODULES[cfg.method](cfg, input_sizes)
