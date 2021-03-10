from config.defaults import Experiment, Dataset, SiMVC, DDC, Fusion, MLP, Loss, CoMVC, Optimizer


rgbd = Experiment(
    dataset_config=Dataset(name="rgbd"),
    model_config=SiMVC(
        backbone_configs=(
            MLP(input_size=(2048,)),
            MLP(input_size=(300,)),
        ),
        fusion_config=Fusion(method="weighted_mean", n_views=2),
        cm_config=DDC(n_clusters=13),
        loss_config=Loss(
            funcs="ddc_1|ddc_2|ddc_3",
        )
    ),
)

rgbd_contrast = Experiment(
    dataset_config=Dataset(name="rgbd"),
    model_config=CoMVC(
        backbone_configs=(
            MLP(input_size=(2048,)),
            MLP(input_size=(300,)),
        ),
        fusion_config=Fusion(method="weighted_mean", n_views=2),
        projector_config=None,
        cm_config=DDC(n_clusters=13),
        loss_config=Loss(
            funcs="ddc_1|ddc_2|ddc_3|contrast",
        ),
        optimizer_config=Optimizer(scheduler_step_size=50, scheduler_gamma=0.5)
    ),
)
