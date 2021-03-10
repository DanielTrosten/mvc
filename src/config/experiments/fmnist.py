from config.defaults import Experiment, SiMVC, CNN, DDC, Fusion, Loss, Dataset, CoMVC, Optimizer

fmnist = Experiment(
    dataset_config=Dataset(name="fmnist"),
    model_config=SiMVC(
        backbone_configs=(
            CNN(input_size=(1, 28, 28)),
            CNN(input_size=(1, 28, 28)),
        ),
        fusion_config=Fusion(method="weighted_mean", n_views=2),
        cm_config=DDC(n_clusters=10),
        loss_config=Loss(
            funcs="ddc_1|ddc_2|ddc_3",
        ),
        optimizer_config=Optimizer()
    ),
)


fmnist_contrast = Experiment(
    dataset_config=Dataset(name="fmnist"),
    model_config=CoMVC(
        backbone_configs=(
            CNN(input_size=(1, 28, 28)),
            CNN(input_size=(1, 28, 28)),
        ),
        fusion_config=Fusion(method="weighted_mean", n_views=2),
        projector_config=None,
        cm_config=DDC(n_clusters=10),
        loss_config=Loss(
            funcs="ddc_1|ddc_2|ddc_3|contrast",
        ),
        optimizer_config=Optimizer()
    ),
)
