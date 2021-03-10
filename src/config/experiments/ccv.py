from config.defaults import Experiment, SiMVC, DDC, Fusion, MLP, Loss, Dataset, CoMVC, Optimizer

ccv = Experiment(
    dataset_config=Dataset(name="ccv"),
    model_config=SiMVC(
        backbone_configs=(
            MLP(input_size=(5000,)),
            MLP(input_size=(5000,)),
            MLP(input_size=(4000,)),
        ),
        fusion_config=Fusion(method="weighted_mean", n_views=3),
        cm_config=DDC(n_clusters=20),
        loss_config=Loss(
            funcs="ddc_1|ddc_2|ddc_3",
        ),
        optimizer_config=Optimizer()
    ),
)

ccv_contrast = Experiment(
    dataset_config=Dataset(name="ccv"),
    model_config=CoMVC(
        backbone_configs=(
            MLP(input_size=(5000,)),
            MLP(input_size=(5000,)),
            MLP(input_size=(4000,)),
        ),
        fusion_config=Fusion(method="weighted_mean", n_views=3),
        projector_config=None,
        cm_config=DDC(n_clusters=20),
        loss_config=Loss(
            funcs="ddc_1|ddc_2|ddc_3|contrast",
            delta=20.0
        ),
        optimizer_config=Optimizer(scheduler_step_size=50, scheduler_gamma=0.1)
    ),
    n_epochs=100
)
