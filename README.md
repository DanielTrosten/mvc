# SiMVC & CoMVC

This repository provides the implementations of SiMVC and CoMVC, presented in the paper:

"Reconsidering Representation Alignment for Multi-view Clustering" by
Daniel J. Trosten, Sigurd Løkse, Robert Jenssen and Michael Kampffmeyer, in _CVPR 2021_.

BibTeX:
```text
@inproceedings{trostenMVC,
  title        = {Reconsidering Representation Alignment for Multi-view Clustering},
  author       = {Daniel J. Trosten and Sigurd Løkse and Robert Jenssen and Michael Kampffmeyer},
  year         = 2021,
  booktitle    = {2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}
}
```


## Installation
Requires Python >= 3.7 (tested on 3.8)

To install the required packages, run:
```
pip install -r requirements.txt
```
from the root directory of the repository. Anaconda (or similar) is recommended.

## Datasets
### Included dataset
The following datasets are included as files in this project:

- `voc` (VOC)
- `rgbd` (RGB-D)  
- `blobs_overlap_5` (Toy dataset with 5 clusters)
- `blobs_overlap` (Toy dataset with 3 clusters)

### Generating datasets
To generate training-ready datasets, run:
```
python -m data.make_dataset <dataset_1> <dataset_2> <...> 
```
This will export the training-ready datasets to `data/processed/<datset_name>.npz`.

Currently, the following datasets can be generated without downloading additional files:

- `mnist_mv` (E-MNIST) 
- `fmnist` (E-FMNIST)

### Datasets that require additional downloads

- `ccv` (CCV): Download the files from [here](https://www.ee.columbia.edu/ln/dvmm/CCV/), and place them in 
`data/raw/CCV`.
- `coil` (COIL-20). Download the processed files from 
[here](https://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php), and place them in `data/raw/COIL`.

After downloading and extracting the files, run
``` Bash
python -m data.make_dataset ccv coil
```
to generate training-ready versions of CCV and COIL-20.

### Preparing a custom dataset for training
Create `<custom_dataset_name>.npz` in `data/processed/` with the following keys:
```
n_views: The number of views, V
labels: One-dimensional array of labels. Shape (n,)
view_0: Data for first view. Shape (n, ...)
  .
  .
  .
view_V: Data for view V. Shape (n, ...)
```
Alternatively, call
```Python
data.make_dataset.export_dataset(
    "<custom_dataset_name>",    # Name of the dataset
    views,                      # List of view-arrays
    labels                      # Label array
)
```
This will automatically export the dataset to an `.npz` file at the correct location.

Then, in the Experiment-config, set
```Python
dataset_config=Dataset("<custom_dataset_name>")
```

## Experiment configuration
Experiment configs are nested configuration objects, where the top-level config is an instance of 
`config.defaults.Experiments`. 

The configuration object for the contrastive model on E-MNIST, for instance, looks like this:
```Python
from config.defaults import Experiment, CNN, DDC, Fusion, Loss, Dataset, CoMVC, Optimizer


mnist_contrast = Experiment(
    dataset_config=Dataset(name="mnist_mv"),
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
            # Additional loss parameters go here
        ),
        optimizer_config=Optimizer(
            learning_rate=1e-3,
            # Additional optimizer parameters go here
        ) 
    ),
    n_epochs=100,
    n_runs=20,
)
```

## Running an experiment
In the `src` directory, run:
```
python -m models.train -c <config_name> 
```
where `<config_name>` is the name of an experiment config from one of the files in `src/config/experiments/` or from 
'src/config/eamc/experiments.py' (for EAMC experiments).

### Overriding config parameters at the command-line
Parameters set in the config object can be overridden at the command line. For instance, if we want to change the 
learning rate for the E-MNIST experiment below from 0.001 to 0.0001, and the number of epochs from 100 to 200,
we can run:
```
python -m models.train -c mnist_contrast \
                       --n_epochs 200 \
                       --model_config__optimizer_config__learning_rate 0.0001
```
Note the double underscores to traverse the hierarchy of the config-object.

## Evaluating an experiment
Run the evaluation script:
```Bash
python -m models.evaluate -c <config_name> \ # Name of the experiment config
                          -t <tag> \         # The unique 8-character ID assigned to the experiment when calling models.train
                          --plot             # Optional flag to plot the representations before and after fusion.
```

## Ablation studies and noise experiment
To run one of these experiments, execute the corresponding script in the `src/scripts` directory.