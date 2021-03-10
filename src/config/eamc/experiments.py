"""
Custom implementation of End-to-End Adversarial-Attention Network for Multi-Modal Clustering (EAMC).
https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhou_End-to-End_Adversarial-Attention_Network_for_Multi-Modal_Clustering_CVPR_2020_paper.pdf
Based on code sent to us by the original authors.
"""

from config.defaults import MLP, CNN, DDC, Dataset
from config.eamc.defaults import EAMCExperiment, EAMC, AttentionLayer, Discriminator, Loss, Optimizer

BACKBONE_MLP_LAYERS = (200, 200, 500)
CNN_LAYERS = (
    ("conv", 5, 5, 32, "relu"),
    ("pool", 2, 2),
    ("conv", 5, 5, 64, "relu"),
    ("pool", 2, 2),
    ("fc", 500),
    ("bn",),
    ("relu",)
)
CNN_BACKBONES = (
    CNN(layers=CNN_LAYERS, input_size=(1, 28, 28)),
    CNN(layers=CNN_LAYERS, input_size=(1, 28, 28)),
)


eamc_blobs_overlap = EAMCExperiment(
    dataset_config=Dataset(name="blobs_overlap"),
    model_config=EAMC(
        backbone_configs=(
            MLP(layers=[32, 32, 32], input_size=(2,)),
            MLP(layers=[32, 32, 32], input_size=(2,)),
        ),
        discriminator_config=Discriminator(
            mlp_config=MLP(layers=(32, 32, 32))
        ),
        loss_config=Loss(),
        cm_config=DDC(n_clusters=3),
        optimizer_config=Optimizer(lr_backbones=2e-4, lr_disc=1e-5)
    ),
)

eamc_blobs_overlap_5 = EAMCExperiment(
    dataset_config=Dataset(name="blobs_overlap_5"),
    model_config=EAMC(
        backbone_configs=(
            MLP(layers=[32, 32, 32], input_size=(2,)),
            MLP(layers=[32, 32, 32], input_size=(2,)),
        ),
        discriminator_config=Discriminator(
            mlp_config=MLP(layers=(32, 32, 32))
        ),
        loss_config=Loss(),
        cm_config=DDC(n_clusters=5),
        optimizer_config=Optimizer(lr_backbones=2e-4, lr_disc=1e-5)
    ),
)

eamc_mnist = EAMCExperiment(
    dataset_config=Dataset(name="mnist_mv"),
    model_config=EAMC(
        backbone_configs=CNN_BACKBONES,
        cm_config=DDC(n_clusters=10),
    ),
)

eamc_mnist_var_noise = EAMCExperiment(
    dataset_config=Dataset(name="mnist_mv", noise_sd=1.0, noise_views=(1,)),
    model_config=EAMC(
        backbone_configs=CNN_BACKBONES,
        cm_config=DDC(n_clusters=10),
    ),
)


eamc_fmnist = EAMCExperiment(
    dataset_config=Dataset(name="fmnist"),
    model_config=EAMC(
        backbone_configs=CNN_BACKBONES,
        cm_config=DDC(n_clusters=10),
    ),
)

eamc_coil = EAMCExperiment(
    dataset_config=Dataset(name="coil"),
    model_config=EAMC(
        backbone_configs=(
            CNN(input_size=(1, 128, 128)),
            CNN(input_size=(1, 128, 128)),
            CNN(input_size=(1, 128, 128)),
        ),
        cm_config=DDC(n_clusters=20),
        attention_config=AttentionLayer(n_views=3)
    ),
)

eamc_rgbd = EAMCExperiment(
    dataset_config=Dataset(name="rgbd"),
    model_config=EAMC(
        backbone_configs=(
            MLP(layers=BACKBONE_MLP_LAYERS, input_size=(2048,)),
            MLP(layers=BACKBONE_MLP_LAYERS, input_size=(300,)),
        ),
        cm_config=DDC(n_clusters=13),
        optimizer_config=Optimizer(lr_backbones=6e-5, lr_disc=2e-5)
    ),
)
