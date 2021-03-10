"""
Custom implementation of End-to-End Adversarial-Attention Network for Multi-Modal Clustering (EAMC).
https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhou_End-to-End_Adversarial-Attention_Network_for_Multi-Modal_Clustering_CVPR_2020_paper.pdf
Based on code sent to us by the original authors.
"""

import torch as th
from torch.nn.functional import binary_cross_entropy

import config
from lib import loss, kernel


class AttLoss(loss.LossTerm):
    """
    Attention loss
    """
    required_tensors = ["backbone_kernels", "fusion_kernel"]

    def __call__(self, net, cfg, extra):
        kc = th.sum(net.weights[None, None, :] * th.stack(extra["backbone_kernels"], dim=-1), dim=-1)
        dif = (extra["fusion_kernel"] - kc)
        return th.trace(dif @ th.t(dif))


class GenLoss(loss.LossTerm):
    """
    Generator loss
    """
    def __call__(self, net, cfg, extra):
        tot = th.tensor(0., device=config.DEVICE)
        target = th.ones(net.output.size(0), device=config.DEVICE)
        for _, dv in net.discriminator_outputs:
            tot += binary_cross_entropy(dv.squeeze(), target)
        return cfg.gamma * tot


class DiscLoss(loss.LossTerm):
    """
    Discriminator loss
    """
    def __call__(self, net, cfg, extra):
        tot = th.tensor(0., device=config.DEVICE)
        real_target = th.ones(net.output.size(0), device=config.DEVICE)
        fake_target = th.zeros(net.output.size(0), device=config.DEVICE)
        for d0, dv in net.discriminator_outputs:
            tot += binary_cross_entropy(dv.squeeze(), fake_target) + binary_cross_entropy(d0.squeeze(), real_target)
        return tot


def backbone_kernels(net, cfg):
    return [kernel.vector_kernel(h, cfg.rel_sigma) for h in net.backbone_outputs]


def fusion_kernel(net, cfg):
    return kernel.vector_kernel(net.fused, cfg.rel_sigma)


class Loss(loss.Loss):
    # Override the TERM_CLASSES and EXTRA_FUNCS of the Loss class, so we can include the EAMC losses.
    TERM_CLASSES = {
        "ddc_1": loss.DDC1,
        "ddc_2_flipped": loss.DDC2Flipped,
        "ddc_2": loss.DDC2,
        "ddc_3": loss.DDC3,
        "att": AttLoss,
        "gen": GenLoss,
        "disc": DiscLoss
    }
    EXTRA_FUNCS = {
        "hidden_kernel": loss.hidden_kernel,
        "backbone_kernels": backbone_kernels,
        "fusion_kernel": fusion_kernel
    }
