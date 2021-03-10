from config.defaults import Experiment, Dataset, SiMVC, DDC, Fusion, MLP, Loss, CoMVC, Optimizer

voc = Experiment(
    dataset_config=Dataset(name="voc"),
    model_config=SiMVC(
        backbone_configs=(
            MLP(input_size=(512,)),
            MLP(input_size=(399,)),
        ),
        fusion_config=Fusion(method="weighted_mean", n_views=2),
        cm_config=DDC(n_clusters=20),
        loss_config=Loss(
            funcs="ddc_1|ddc_2|ddc_3",
        ),
        optimizer_config=Optimizer(learning_rate=1e-3, scheduler_step_size=50, scheduler_gamma=0.1)
    ),
)

voc_contrast = Experiment(
    dataset_config=Dataset(name="voc"),
    model_config=CoMVC(
        backbone_configs=(
            MLP(input_size=(512,)),
            MLP(input_size=(399,)),
        ),
        projector_config=None,
        fusion_config=Fusion(method="weighted_mean", n_views=2),
        cm_config=DDC(n_clusters=20),
        loss_config=Loss(
            funcs="ddc_1|ddc_2|ddc_3|contrast",
        ),
        optimizer_config=Optimizer(learning_rate=1e-3)
    ),
)
