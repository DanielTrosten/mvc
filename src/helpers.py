import math
import pandas as pd
import numpy as np
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

import config


def npy(t, to_cpu=True):
    """
    Convert a tensor to a numpy array.

    :param t: Input tensor
    :type t: th.Tensor
    :param to_cpu: Call the .cpu() method on `t`?
    :type to_cpu: bool
    :return: Numpy array
    :rtype: np.ndarray
    """
    if isinstance(t, (list, tuple)):
        # We got a list. Convert each element to numpy
        return [npy(ti) for ti in t]
    elif isinstance(t, dict):
        # We got a dict. Convert each value to numpy
        return {k: npy(v) for k, v in t.items()}
    # Assuming t is a tensor.
    if to_cpu:
        return t.cpu().detach().numpy()
    return t.detach().numpy()


def ensure_iterable(elem, expected_length=1):
    if isinstance(elem, (list, tuple)):
        assert len(elem) == expected_length, f"Expected iterable {elem} with length {len(elem)} does not have " \
                                             f"expected length {expected_length}"
    else:
        elem = expected_length * [elem]
    return elem


def dict_means(dicts):
    """
    Compute the mean value of keys in a list of dicts

    :param dicts: Input dicts
    :type dicts: List[dict]
    :return: Mean values
    :rtype: dict
    """
    return pd.DataFrame(dicts).mean(axis=0).to_dict()


def add_prefix(dct, prefix, sep="/"):
    """
    Add a prefix to all keys in `dct`.

    :param dct: Input dict
    :type dct: dict
    :param prefix: Prefix
    :type prefix: str
    :param sep: Separator between prefix and key
    :type sep: str
    :return: Dict with prefix prepended to all keys
    :rtype: dict
    """
    return {prefix + sep + key: value for key, value in dct.items()}


def ordered_cmat(labels, pred):
    """
    Compute the confusion matrix and accuracy corresponding to the best cluster-to-class assignment.

    :param labels: Label array
    :type labels: np.array
    :param pred: Predictions array
    :type pred: np.array
    :return: Accuracy and confusion matrix
    :rtype: Tuple[float, np.array]
    """
    cmat = confusion_matrix(labels, pred)
    ri, ci = linear_sum_assignment(-cmat)
    ordered = cmat[np.ix_(ri, ci)]
    acc = np.sum(np.diag(ordered))/np.sum(ordered)
    return acc, ordered


def get_save_dir(experiment_name, identifier, run):
    """
    Get the save dir for an experiment

    :param experiment_name: Name of the config
    :type experiment_name: str
    :param identifier: 8-character unique identifier for the current experiment
    :type identifier: str
    :param run: Current training run
    :type run: int
    :return: Path to save dir
    :rtype: pathlib.Path
    """
    if not str(run).startswith("run-"):
        run = f"run-{run}"
    return config.MODELS_DIR / f"{experiment_name}-{identifier}" / run


def he_init_weights(module):
    """
    Initialize network weights using the He (Kaiming) initialization strategy.

    :param module: Network module
    :type module: nn.Module
    """
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight)


def num2tuple(num):
    return num if isinstance(num, (tuple, list)) else (num, num)


def conv2d_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Compute the output shape of a convolution operation.

    :param h_w: Height and width of input
    :type h_w: Tuple[int, int]
    :param kernel_size: Size of kernel
    :type kernel_size: Union[int, Tuple[int, int]]
    :param stride: Stride of convolution
    :type stride: Union[int, Tuple[int, int]]
    :param pad: Padding (in pixels)
    :type pad: Union[int, Tuple[int, int]]
    :param dilation: Dilation
    :type dilation: Union[int, Tuple[int, int]]
    :return: Height and width of output
    :rtype: Tuple[int, int]
    """
    h_w, kernel_size, stride, = num2tuple(h_w), num2tuple(kernel_size), num2tuple(stride)
    pad, dilation = num2tuple(pad), num2tuple(dilation)

    h = math.floor((h_w[0] + 2 * pad[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
    w = math.floor((h_w[1] + 2 * pad[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)
    return h, w
