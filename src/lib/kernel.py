import torch as th
from torch.nn.functional import relu


EPSILON = 1E-9


def kernel_from_distance_matrix(dist, rel_sigma, min_sigma=EPSILON):
    """
    Compute a Gaussian kernel matrix from a distance matrix.

    :param dist: Disatance matrix
    :type dist: th.Tensor
    :param rel_sigma: Multiplication factor for the sigma hyperparameter
    :type rel_sigma: float
    :param min_sigma: Minimum value for sigma. For numerical stability.
    :type min_sigma: float
    :return: Kernel matrix
    :rtype: th.Tensor
    """
    # `dist` can sometimes contain negative values due to floating point errors, so just set these to zero.
    dist = relu(dist)
    sigma2 = rel_sigma * th.median(dist)
    # Disable gradient for sigma
    sigma2 = sigma2.detach()
    sigma2 = th.where(sigma2 < min_sigma, sigma2.new_tensor(min_sigma), sigma2)
    k = th.exp(- dist / (2 * sigma2))
    return k


def vector_kernel(x, rel_sigma=0.15):
    """
    Compute a kernel matrix from the rows of a matrix.

    :param x: Input matrix
    :type x: th.Tensor
    :param rel_sigma: Multiplication factor for the sigma hyperparameter
    :type rel_sigma: float
    :return: Kernel matrix
    :rtype: th.Tensor
    """
    return kernel_from_distance_matrix(cdist(x, x), rel_sigma)


def cdist(X, Y):
    """
    Pairwise distance between rows of X and rows of Y.

    :param X: First input matrix
    :type X: th.Tensor
    :param Y: Second input matrix
    :type Y: th.Tensor
    :return: Matrix containing pairwise distances between rows of X and rows of Y
    :rtype: th.Tensor
    """
    xyT = X @ th.t(Y)
    x2 = th.sum(X**2, dim=1, keepdim=True)
    y2 = th.sum(Y**2, dim=1, keepdim=True)
    d = x2 - 2 * xyT + th.t(y2)
    return d
