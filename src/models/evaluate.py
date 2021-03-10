import os
import argparse
import numpy as np
import torch as th
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from sklearn.metrics import normalized_mutual_info_score

import helpers
from models.build_model import from_file

IGNORE_IN_TOTAL = ("contrast",)


def calc_metrics(labels, pred):
    """
    Compute metrics.

    :param labels: Label tensor
    :type labels: th.Tensor
    :param pred: Predictions tensor
    :type pred: th.Tensor
    :return: Dictionary containing calculated metrics
    :rtype: dict
    """
    acc, cmat = helpers.ordered_cmat(labels, pred)
    metrics = {
        "acc": acc,
        "cmat": cmat,
        "nmi": normalized_mutual_info_score(labels, pred, average_method="geometric"),
    }
    return metrics


def get_log_params(net):
    """
    Get the network parameters we want to log.

    :param net: Model
    :type net:
    :return:
    :rtype:
    """
    params_dict = {}
    weights = []
    if getattr(net, "fusion", None) is not None:
        with th.no_grad():
            weights = net.fusion.get_weights(softmax=True)

    elif hasattr(net, "attention"):
        weights = net.weights

    for i, w in enumerate(helpers.npy(weights)):
        params_dict[f"fusion/weight_{i}"] = w

    if hasattr(net, "discriminators"):
        for i, discriminator in enumerate(net.discriminators):
            d0, dv = helpers.npy([discriminator.d0, discriminator.dv])
            params_dict[f"discriminator_{i}/d0/mean"] = d0.mean()
            params_dict[f"discriminator_{i}/d0/std"] = d0.std()
            params_dict[f"discriminator_{i}/dv/mean"] = dv.mean()
            params_dict[f"discriminator_{i}/dv/std"] = dv.std()

    return params_dict


def get_eval_data(dataset, n_eval_samples, batch_size):
    """
    Create a dataloader to use for evaluation

    :param dataset: Inout dataset.
    :type dataset: th.utils.data.Dataset
    :param n_eval_samples: Number of samples to include in the evaluation dataset. Set to None to use all available
                           samples.
    :type n_eval_samples: int
    :param batch_size: Batch size used for training.
    :type batch_size: int
    :return: Evaluation dataset loader
    :rtype: th.utils.data.DataLoader
    """
    if n_eval_samples is not None:
        *views, labels = dataset.tensors
        n = views[0].size(0)
        idx = np.random.choice(n, min(n, n_eval_samples), replace=False)
        views, labels = [v[idx] for v in views], labels[idx]
        dataset = th.utils.data.TensorDataset(*views, labels)

    eval_loader = th.utils.data.DataLoader(dataset, batch_size=int(batch_size), shuffle=True, num_workers=0,
                                           drop_last=False, pin_memory=False)
    return eval_loader


def batch_predict(net, eval_data, batch_size):
    """
    Compute predictions for `eval_data` in batches. Batching does not influence predictions, but it influences the loss
    computations.

    :param net: Model
    :type net:
    :param eval_data: Evaluation dataloader
    :type eval_data: th.utils.data.DataLoader
    :param batch_size: Batch size
    :type batch_size: int
    :return: Label tensor, predictions tensor, list of dicts with loss values, array containing mean and std of cluster
             sizes.
    :rtype:
    """
    predictions = []
    labels = []
    losses = []
    cluster_sizes = []

    net.eval()
    with th.no_grad():
        for i, (*batch, label) in enumerate(eval_data):
            pred = net(batch)
            labels.append(helpers.npy(label))
            predictions.append(helpers.npy(pred).argmax(axis=1))

            # Only calculate losses for full batches
            if label.size(0) == batch_size:
                batch_losses = net.calc_losses(ignore_in_total=IGNORE_IN_TOTAL)
                losses.append(helpers.npy(batch_losses))
                cluster_sizes.append(helpers.npy(pred.sum(dim=0)))

    labels = np.concatenate(labels, axis=0)
    predictions = np.concatenate(predictions, axis=0)
    net.train()
    return labels, predictions, losses, np.array(cluster_sizes).sum(axis=0)


def get_logs(cfg, net, eval_data, iter_losses=None, epoch=None, include_params=True):
    if iter_losses is not None:
        logs = helpers.add_prefix(helpers.dict_means(iter_losses), "iter_losses")
    else:
        logs = {}
    if (epoch is None) or ((epoch % cfg.eval_interval) == 0):
        labels, pred, eval_losses, cluster_sizes = batch_predict(net, eval_data, cfg.batch_size)
        eval_losses = helpers.dict_means(eval_losses)
        logs.update(helpers.add_prefix(eval_losses, "eval_losses"))
        logs.update(helpers.add_prefix(calc_metrics(labels, pred), "metrics"))
        logs.update(helpers.add_prefix({"mean": cluster_sizes.mean(), "sd": cluster_sizes.std()}, "cluster_size"))
    if include_params:
        logs.update(helpers.add_prefix(get_log_params(net), "params"))
    if epoch is not None:
        logs["epoch"] = epoch
    return logs


def eval_run(cfg, cfg_name, experiment_identifier, run, net, eval_data, callbacks=tuple(), load_best=True):
    """
    Evaluate a training run.

    :param cfg: Experiment config
    :type cfg: config.defaults.Experiment
    :param cfg_name: Config name
    :type cfg_name: str
    :param experiment_identifier: 8-character unique identifier for the current experiment
    :type experiment_identifier: str
    :param run: Run to evaluate
    :type run: int
    :param net: Model
    :type net:
    :param eval_data: Evaluation dataloder
    :type eval_data: th.utils.data.DataLoader
    :param callbacks: List of callbacks to call after evaluation
    :type callbacks: List
    :param load_best: Load the "best.pt" model before evaluation?
    :type load_best: bool
    :return: Evaluation logs
    :rtype: dict
    """
    if load_best:
        model_path = helpers.get_save_dir(cfg_name, experiment_identifier, run) / "best.pt"
        if os.path.isfile(model_path):
            net.load_state_dict(th.load(model_path))
        else:
            print(f"Unable to load best model for evaluation. Model file not found: {model_path}")
    logs = get_logs(cfg, net, eval_data, include_params=True)
    for cb in callbacks:
        cb.at_eval(net=net, logs=logs)
    return logs


def eval_experiment(cfg_name, tag, plot=False):
    """
    Evaluate a full experiment

    :param cfg_name: Name of the config
    :type cfg_name: str
    :param tag: 8-character unique identifier for the current experiment
    :type tag: str
    :param plot: Display a scatterplot of the representations before and after fusion?
    :type plot: bool
    """
    max_n_runs = 100
    best_logs = None
    best_run = None
    best_net = None
    best_loss = np.inf

    for run in range(max_n_runs):
        try:
            net, views, labels, cfg = from_file(cfg_name, tag, run, ckpt="best", return_data=True, return_config=True)
        except FileNotFoundError:
            break

        eval_dataset = th.utils.data.TensorDataset(*[th.tensor(v) for v in views], th.tensor(labels))
        eval_data = get_eval_data(eval_dataset, cfg.n_eval_samples, cfg.batch_size)
        run_logs = eval_run(cfg, cfg_name, tag, run, net, eval_data, load_best=False)
        del run_logs["metrics/cmat"]

        if run_logs[f"eval_losses/{cfg.best_loss_term}"] < best_loss:
            best_loss = run_logs[f"eval_losses/{cfg.best_loss_term}"]
            best_logs = run_logs
            best_run = run
            best_net = net

    print(f"\nBest run was {best_run}.", end="\n\n")
    headers = ["Name", "Value"]
    values = list(best_logs.items())
    print(tabulate(values, headers=headers), "\n")
    
    if plot:
        plot_representations(views, labels, best_net)
        plt.show()
    

def plot_representations(views, labels, net, project_method="pca"):
    with th.no_grad():
        output = net([th.tensor(v) for v in views])
        pred = helpers.npy(output).argmax(axis=1)

        hidden = helpers.npy(net.backbone_outputs)
        fused = helpers.npy(net.fused)

    hidden = np.concatenate(hidden, axis=0)
    view_hue = sum([labels.shape[0] * [str(i + 1)] for i in range(2)], [])
    fused_hue = [str(l + 1) for l in pred]

    view_cmap = "tab10"
    class_cmap = "hls"
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    plot_projection(X=hidden, method=project_method, hue=view_hue, ax=ax[0], title="Before fusion",
                    legend_title="View", hue_order=sorted(list(set(view_hue))), cmap=view_cmap)
    plot_projection(X=fused, method=project_method, hue=fused_hue, ax=ax[1], title="After fusion",
                    legend_title="Prediction", hue_order=sorted(list(set(fused_hue))), cmap=class_cmap)


def plot_projection(X, method, hue, ax, title=None, cmap="tab10", legend_title=None, legend_loc=1, **kwargs):
    X = project(X, method)
    pl = sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=hue, ax=ax, legend="full", palette=cmap, **kwargs)
    leg = pl.get_legend()
    leg._loc = legend_loc
    if title is not None:
        ax.set_title(title)
    if legend_title is not None:
        leg.set_title(legend_title)


def project(X, method):
    if method == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=2).fit_transform(X)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        return TSNE(n_components=2).fit_transform(X)
    elif method is None:
        return X
    else:
        raise RuntimeError()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", dest="cfg_name", required=True)
    parser.add_argument("-t", "--tag", dest="tag", required=True)
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    eval_experiment(args.cfg_name, args.tag, args.plot)
