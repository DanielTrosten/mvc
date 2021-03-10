import torch as th

import config
import helpers
from models.ddc import DDCModel
from models.simple_mvc import SiMVC
from models.contrastive_mvc import CoMVC
from eamc.model import EAMC
from data.load import load_dataset


MODEL_CONSTRUCTORS = {
    "DDCModel": DDCModel,
    "SiMVC": SiMVC,
    "CoMVC": CoMVC,
    "EAMC": EAMC
}


def build_model(model_cfg):
    """
    Build the model specified by `model_cfg`.

    :param model_cfg: Config of model to build
    :type model_cfg: Union[config.defaults.DDCModel, config.defaults.SiMVC, config.defaults.CoMVC,
                           config.eamc.defaults.EAMC]
    :return: Model
    :rtype: Union[DDCModel, SiMVC, CoMVC, EAMC]
    """
    if model_cfg.class_name not in MODEL_CONSTRUCTORS:
        raise ValueError(f"Invalid model type: {model_cfg.type}")
    model = MODEL_CONSTRUCTORS[model_cfg.class_name](model_cfg).to(config.DEVICE, non_blocking=True)
    return model


def from_file(experiment_name=None, tag=None, run=None, ckpt="best", return_data=False, return_config=False, **kwargs):
    """
    Load a trained from disc

    :param experiment_name: Name of the experiment (name of the config)
    :type experiment_name: str
    :param tag: 8-character experiment identifier
    :type tag: str
    :param run: Training run to load
    :type run: int
    :param ckpt: Checkpoint to load. Specify a valid checkpoint, or "best" to load the best model.
    :type ckpt: Union[int, str]
    :param return_data: Return the dataset?
    :type return_data: bool
    :param return_config: Return the experiment config?
    :type return_config: bool
    :param kwargs:
    :type kwargs:
    :return: Loaded model, dataset (if return_data == True), config (if return_config == True)
    :rtype:
    """
    try:
        cfg = config.get_config_from_file(name=experiment_name, tag=tag)
    except FileNotFoundError:
        print("WARNING: Could not get pickled config.")
        cfg = config.get_config_by_name(experiment_name)

    model_dir = helpers.get_save_dir(experiment_name, identifier=tag, run=run)
    if ckpt == "best":
        model_file = "best.pt"
    else:
        model_file = f"checkpoint_{str(ckpt).zfill(4)}.pt"

    model_path = model_dir / model_file
    net = build_model(cfg.model_config)
    print(f"Loading model from {model_path}")
    net.load_state_dict(th.load(model_path, map_location=config.DEVICE))
    net.eval()

    out = [net]

    if return_data:
        dataset_kwargs = cfg.dataset_config.dict()
        for key, value in kwargs.items():
            dataset_kwargs[key] = value
        views, labels = load_dataset(to_dataset=False, **dataset_kwargs)
        out = [net, views, labels]

    if return_config:
        out.append(cfg)

    if len(out) == 1:
        out = out[0]

    return out
