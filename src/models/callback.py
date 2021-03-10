import os
import yaml
import pickle
import numpy as np
import torch as th
from tabulate import tabulate

import helpers


class Callback:
    def __init__(self, epoch_interval=1, batch_interval=1):
        """
        Base class for training callbacks

        :param epoch_interval: Number of epochs between calling `at_epoch_end`.
        :type epoch_interval: int
        :param batch_interval: Number of batches between calling `at_batch_end`.
        :type batch_interval: int
        """
        self.epoch_interval = epoch_interval
        self.batch_interval = batch_interval

    def epoch_end(self, epoch, **kwargs):
        if not (epoch % self.epoch_interval):
            return self.at_epoch_end(epoch, **kwargs)

    def batch_end(self, epoch, batch, **kwargs):
        if (not (epoch % self.epoch_interval)) and (not (batch % self.batch_interval)):
            return self.at_batch_end(epoch, batch, **kwargs)

    def at_epoch_end(self, epoch, logs=None, net=None, **kwargs):
        pass

    def at_batch_end(self, epoch, batch, outputs=None, losses=None, net=None, **kwargs):
        pass

    def at_eval(self, net=None, logs=None, **kwargs):
        pass


class Printer(Callback):
    def __init__(self, print_confusion_matrix=True, **kwargs):
        """
        Print logs to the terminal.

        :param print_confusion_matrix: Print the confusion matrix when it is available?
        :type print_confusion_matrix: bool
        :param kwargs:
        :type kwargs:
        """
        super().__init__(**kwargs)
        self.ignore_keys = ["iter_losses/"]
        if not print_confusion_matrix:
            self.ignore_keys.append("metrics/cmat")
            
        np.set_printoptions(edgeitems=20, linewidth=200)

    def at_epoch_end(self, epoch, logs=None, net=None, **kwargs):
        print_logs = logs.copy()
        for key in logs.keys():
            if any([key.startswith(ik) for ik in self.ignore_keys]):
                del print_logs[key]

        headers = ["Key", "Value"]
        values = list(print_logs.items())
        print(tabulate(values, headers=headers), "\n")


class ModelSaver(Callback):
    def __init__(self, cfg, experiment_name, identifier, run, best_loss_term, checkpoint_interval=1, **kwargs):
        """
        Model saver callback. Saves model at specified checkpoints, or when `best_loss_term` in the loss function
        reaches the lowest observed value.

        :param cfg: Experiment config
        :type cfg: config.defaults.Experiment
        :param experiment_name: Name of the experiment
        :type experiment_name: str
        :param identifier: 8-character unique experiment identifier
        :type identifier: str
        :param run: Current training run
        :type run: int
        :param best_loss_term: Term in the loss function to monitor.
        :type best_loss_term: str
        :param checkpoint_interval: Number of epochs between saving model checkpoints.
        :type checkpoint_interval: int
        :param kwargs:
        :type kwargs:
        """
        super().__init__(**kwargs)

        self.best_loss_term = f"eval_losses/{best_loss_term}"
        self.min_loss = np.inf
        self.checkpoint_interval = checkpoint_interval
        self.save_dir = helpers.get_save_dir(experiment_name, identifier, run)
        os.makedirs(self.save_dir, exist_ok=True)
        self._save_cfg(cfg)

    def _save_cfg(self, cfg):
        with open(self.save_dir / "config.yml", "w") as f:
            yaml.dump(cfg.dict(), f)
        with open(self.save_dir / "config.pkl", "wb") as f:
            pickle.dump(cfg, f)

    def _save_model(self, file_name, net):
        model_path = self.save_dir / file_name
        th.save(net.state_dict(), model_path)
        print(f"Model successfully saved: {model_path}")

    def at_epoch_end(self, epoch, outputs=None, logs=None, net=None, **kwargs):
        if not (epoch % self.checkpoint_interval):
            # Save model checkpoint
            self._save_model(f"checkpoint_{str(epoch).zfill(4)}.pt", net)

        avg_loss = logs.get(self.best_loss_term, np.inf)
        # Save to model_best if the current loss is the lowest loss encountered
        if avg_loss < self.min_loss:
            self.min_loss = avg_loss
            self._save_model("best.pt", net)


class StopTraining(Exception):
    pass


class EarlyStopping(Callback):
    def __init__(self, patience, best_loss_term, **kwargs):
        """
        Early stopping callback. Raises a `StopTraining` exception when the term `best_loss_term` in the loss function
        has not decreased in `patience` epochs.

        :param patience: Number of epochs to wait for loss decrease
        :type patience: int
        :param best_loss_term: Term in the loss function to monitor.
        :type best_loss_term: str
        :param kwargs:
        :type kwargs:
        """
        super().__init__(**kwargs)
        self.best_loss_term = f"eval_losses/{best_loss_term}"
        self.patience = patience
        self.min_loss = np.inf
        self.best_epoch = 0

    def at_epoch_end(self, epoch, outputs=None, logs=None, net=None, **kwargs):
        avg_loss = logs.get(self.best_loss_term, np.inf)

        if np.isnan(avg_loss):
            raise StopTraining(f"Got loss = NaN. Training stopped.")

        if avg_loss < self.min_loss:
            self.min_loss = avg_loss
            self.best_epoch = epoch

        if (epoch - self.best_epoch) >= self.patience:
            raise StopTraining(f"Loss has not decreased in {self.patience} epochs. Min. loss was {self.min_loss}. "
                               f"Training stopped.")

