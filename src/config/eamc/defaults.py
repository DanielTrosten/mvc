"""
Custom implementation of End-to-End Adversarial-Attention Network for Multi-Modal Clustering (EAMC).
https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhou_End-to-End_Adversarial-Attention_Network_for_Multi-Modal_Clustering_CVPR_2020_paper.pdf
Based on code sent to us by the original authors.
"""

from typing import Tuple, Union, Optional

from config.config import Config
from config.defaults import MLP, DDC, CNN, Dataset, Fusion


class Loss(Config):
    # Multiplication factor for the sigma hyperparameter
    rel_sigma: float = 0.15
    # Weight of adversarial losses
    gamma: float = 10
    # Number of clusters
    n_clusters: int = None
    # Optional weights for the loss terms. Set to None to have all weights equal to 1.
    weights: Tuple[Union[float, int], ...] = None
    # Terms to use in the loss, separated by '|'. E.g. "ddc_1|ddc_2|ddc_3|" for the DDC clustering loss
    funcs = "ddc_1|ddc_2_flipped|ddc_3|att|gen|disc"


class AttentionLayer(Config):
    # Softmax temperature
    tau: float = 10.0
    # Config for the attention net. Final layer will be added automatically
    mlp_config: MLP = MLP(
        layers=(100, 50),
        activation=None
    )
    # Number of input views
    n_views: int = 2


class Discriminator(Config):
    # Config for the discriminator
    mlp_config: MLP = MLP(
        layers=(256, 256, 128),
        activation="leaky_relu:0.2"
    )


class Optimizer(Config):
    # Discriminator learning rate
    lr_disc: float = 1e-3
    # Encoder learning rate
    lr_backbones: float = 1e-5
    # Attention learning rate
    lr_att: float = 1e-4
    # Clustering module learning rate
    lr_clustering_module: float = 1e-5
    # Beta parameters for the discriminator
    betas_disc = (0.5, 0.999)
    # Beta parameters for the encoders
    betas_backbones = (0.95, 0.999)
    # Beta parameters for the attention net
    betas_att = (0.95, 0.999)
    # Beta parameters for the clustering module
    betas_clustering_module = (0.95, 0.999)


class EAMC(Config):
    # Encoder configs
    backbone_configs: Tuple[Union[MLP, CNN], ...]
    # Attention net config. Set to None to remove attention net
    attention_config: Optional[AttentionLayer] = AttentionLayer()
    # Optional fusion config to use instead of attention net.
    fusion_config: Fusion = None
    # Discriminator config
    discriminator_config: Optional[Discriminator] = Discriminator()
    # Clustering module config
    cm_config: DDC
    # Loss config
    loss_config: Loss = Loss()
    # Optimizer config
    optimizer_config: Optimizer = Optimizer()
    # Max norm for gradient cliping
    clip_norm = 0.5
    # Number of consecutive batches to train the encoders, attention net and clustering module.
    t: int = 1
    # Number of consecutive batches to train the discriminator.
    t_disc: int = 1


class EAMCExperiment(Config):
    # Dataset config
    dataset_config: Dataset
    # Model config
    model_config: EAMC
    # Number of training runs
    n_runs = 20
    # Number of epochs per run
    n_epochs = 500
    # Batch size
    batch_size = 100
    # Number of epochs between model evaluation.
    eval_interval: int = 5
    # Number of epochs between model checkpoints.
    checkpoint_interval = 50
    # Number of samples to use for evaluation. Set to None to use all samples in the dataset.
    n_eval_samples: int = None
    # Patience for early stopping.
    patience: int = 1e9
    # Term in loss function to use for model selection. Set to "tot" to use the sum of all terms.
    best_loss_term = "tot"
