from config.defaults import Experiment, Dataset, SiMVC, MLP, DDC, Fusion, Loss, CoMVC


blobs_overlap = Experiment(
    dataset_config=Dataset(name="blobs_overlap"),
    model_config=SiMVC(
        backbone_configs=(
            MLP(layers=[32, 32, 32], input_size=(2,)),
            MLP(layers=[32, 32, 32], input_size=(2,)),
        ),
        fusion_config=Fusion(method="weighted_mean", n_views=2),
        cm_config=DDC(n_clusters=3),
        loss_config=Loss(
            funcs="ddc_1|ddc_2|ddc_3",
        ),
    ),
    n_runs=1,
    n_epochs=10,
)


blobs_overlap_contrast = Experiment(
    dataset_config=Dataset(name="blobs_overlap"),
    model_config=CoMVC(
        backbone_configs=(
            MLP(layers=[32, 32, 32], input_size=(2,)),
            MLP(layers=[32, 32, 32], input_size=(2,)),
        ),
        fusion_config=Fusion(method="weighted_mean", n_views=2),
        projector_config=None,
        cm_config=DDC(n_clusters=3),
        loss_config=Loss(
            funcs="ddc_1|ddc_2|ddc_3|contrast",
        )
    ),
    n_runs=1,
)

blobs_overlap_5 = Experiment(
    dataset_config=Dataset(name="blobs_overlap_5"),
    model_config=SiMVC(
        backbone_configs=(
            MLP(layers=[32, 32, 32], input_size=(2,)),
            MLP(layers=[32, 32, 32], input_size=(2,)),
        ),
        fusion_config=Fusion(method="weighted_mean", n_views=2),
        cm_config=DDC(n_clusters=5),
        loss_config=Loss(
            funcs="ddc_1|ddc_2|ddc_3",
        ),
    ),
    n_runs=1,
)

blobs_overlap_5_contrast = Experiment(
    dataset_config=Dataset(name="blobs_overlap_5"),
    model_config=CoMVC(
        backbone_configs=(
            MLP(layers=[32, 32, 32], input_size=(2,)),
            MLP(layers=[32, 32, 32], input_size=(2,)),
        ),
        fusion_config=Fusion(method="weighted_mean", n_views=2),
        projector_config=None,
        cm_config=DDC(n_clusters=5),
        loss_config=Loss(
            funcs="ddc_1|ddc_2|ddc_3|contrast",
        )
    ),
    n_runs=1,
)
