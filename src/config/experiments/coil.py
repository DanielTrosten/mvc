from config.defaults import Experiment, SiMVC, CNN, DDC, Fusion, Loss, Dataset, CoMVC, Optimizer


coil = Experiment(
    dataset_config=Dataset(name="coil"),
    model_config=SiMVC(
        backbone_configs=(
            CNN(input_size=(1, 128, 128)),
            CNN(input_size=(1, 128, 128)),
            CNN(input_size=(1, 128, 128)),
        ),
        fusion_config=Fusion(method="weighted_mean", n_views=3),
        cm_config=DDC(n_clusters=20),
        loss_config=Loss(
            funcs="ddc_1|ddc_2|ddc_3",
        ),
        optimizer_config=Optimizer()
    ),
    n_epochs=100,
)

coil_contrast = Experiment(
    dataset_config=Dataset(name="coil"),
    model_config=CoMVC(
        backbone_configs=(
            CNN(input_size=(1, 128, 128)),
            CNN(input_size=(1, 128, 128)),
            CNN(input_size=(1, 128, 128)),
        ),
        fusion_config=Fusion(method="weighted_mean", n_views=3),
        projector_config=None,
        cm_config=DDC(n_clusters=20),
        loss_config=Loss(
            funcs="ddc_1|ddc_2|ddc_3|contrast",
            delta=20.0
        ),
        optimizer_config=Optimizer()
    ),
)