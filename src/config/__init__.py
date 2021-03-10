import argparse
import pickle

from .constants import *
from .config import Config
from . import defaults, experiments
from .eamc import experiments as eamc_experiments
from .eamc import defaults as eamc_defaults


def parse_config_name_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", dest="config_name", required=True)
    return parser.parse_known_args()[0].config_name


def set_cfg_value(cfg, key_list, value):
    sub_cfg = cfg
    for key in key_list[:-1]:
        sub_cfg = getattr(sub_cfg, key)
    setattr(sub_cfg, key_list[-1], value)


def update_cfg(cfg):
    sep = "__"
    parser = argparse.ArgumentParser()
    cfg_dict = hparams_dict(cfg, sep=sep)

    parser.add_argument("-c", "--config", dest="config_name")

    for key, value in cfg_dict.items():
        value_type = type(value) if isinstance(value, (int, float, bool)) else None
        parser.add_argument("--" + key, dest=key, default=value, type=value_type)

    args = parser.parse_args()

    for key in cfg_dict.keys():
        key_list = key.split(sep)
        value = getattr(args, key)
        set_cfg_value(cfg, key_list, value)


def get_config_by_name(name):
    try:
        if name.startswith("eamc"):
            cfg = getattr(eamc_experiments, name)
        else:
            cfg = getattr(experiments, name)
    except Exception as err:
        raise RuntimeError(f"Config not found: {name}") from err
    cfg.model_config.loss_config.n_clusters = cfg.model_config.cm_config.n_clusters
    return cfg


def get_config_from_file(name=None, tag=None, file_path=None, run=0):
    if file_path is None:
        file_path = MODELS_DIR / f"{name}-{tag}" / f"run-{run}" / "config.pkl"
    with open(file_path, "rb") as f:
        cfg = pickle.load(f)
    return cfg


def get_experiment_config():
    name = parse_config_name_arg()
    cfg = get_config_by_name(name)
    update_cfg(cfg)
    return name, cfg


def _insert_hparams(cfg_dict, hp_dict, key_prefix, skip_keys, sep="/"):
    hparam_types = (str, int, float, bool)
    for key, value in cfg_dict.items():
        if key in skip_keys:
            continue
        _key = f"{key_prefix}{sep}{key}" if key_prefix else key
        if isinstance(value, hparam_types) or value is None:
            hp_dict[_key] = value
        elif isinstance(value, dict):
            _insert_hparams(value, hp_dict, _key, skip_keys, sep=sep)


def hparams_dict(cfg, sep="/"):
    skip_keys = []
    hp_dict = {}
    _insert_hparams(cfg.dict(), hp_dict, "", skip_keys, sep=sep)
    return hp_dict
