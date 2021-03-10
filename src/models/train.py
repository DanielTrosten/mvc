import wandb
import torch as th

import config
import helpers
from data.load import load_dataset
from models import callback
from models.build_model import build_model
from models import evaluate


def train(cfg, net, loader, eval_data, callbacks=tuple()):
    """
    Train the model for one run.

    :param cfg: Experiment config
    :type cfg: config.defaults.Experiment
    :param net: Model
    :type net:
    :param loader: DataLoder for training data
    :type loader:  th.utils.data.DataLoader
    :param eval_data: DataLoder for evaluation data
    :type eval_data:  th.utils.data.DataLoader
    :param callbacks: Training callbacks.
    :type callbacks: List
    :return: None
    :rtype: None
    """
    n_batches = len(loader)
    for e in range(1, cfg.n_epochs + 1):
        iter_losses = []
        for i, data in enumerate(loader):
            *batch, _ = data
            try:
                batch_losses = net.train_step(batch, epoch=(e-1), it=i, n_batches=n_batches)
            except Exception as e:
                print(f"Training stopped due to exception: {e}")
                return

            iter_losses.append(helpers.npy(batch_losses))
        logs = evaluate.get_logs(cfg, net, eval_data=eval_data, iter_losses=iter_losses, epoch=e, include_params=True)
        try:
            for cb in callbacks:
                cb.epoch_end(e, logs=logs, net=net)
        except callback.StopTraining as err:
            print(err)
            break


def main():
    """
    Run an experiment.
    """
    experiment_name, cfg = config.get_experiment_config()
    dataset = load_dataset(**cfg.dataset_config.dict())
    loader = th.utils.data.DataLoader(dataset, batch_size=int(cfg.batch_size), shuffle=True, num_workers=0,
                                      drop_last=True, pin_memory=False)
    eval_data = evaluate.get_eval_data(dataset, cfg.n_eval_samples, cfg.batch_size)
    experiment_identifier = wandb.util.generate_id()

    run_logs = []
    for run in range(cfg.n_runs):
        net = build_model(cfg.model_config)
        print(net)
        callbacks = (
            callback.Printer(print_confusion_matrix=(cfg.model_config.cm_config.n_clusters <= 100)),
            callback.ModelSaver(cfg=cfg, experiment_name=experiment_name, identifier=experiment_identifier,
                                run=run, epoch_interval=1, best_loss_term=cfg.best_loss_term,
                                checkpoint_interval=cfg.checkpoint_interval),
            callback.EarlyStopping(patience=cfg.patience, best_loss_term=cfg.best_loss_term, epoch_interval=1)
        )
        train(cfg, net, loader, eval_data=eval_data, callbacks=callbacks)
        run_logs.append(evaluate.eval_run(cfg=cfg, cfg_name=experiment_name,
                                          experiment_identifier=experiment_identifier, run=run, net=net,
                                          eval_data=eval_data, callbacks=callbacks, load_best=True))


if __name__ == '__main__':
    main()
