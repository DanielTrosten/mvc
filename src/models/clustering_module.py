import torch.nn as nn


class DDC(nn.Module):
    def __init__(self, input_dim, cfg):
        """
        DDC clustering module

        :param input_dim: Shape of inputs.
        :param cfg: DDC config. See `config.defaults.DDC`
        """
        super().__init__()

        hidden_layers = [nn.Linear(input_dim[0], cfg.n_hidden), nn.ReLU()]
        if cfg.use_bn:
            hidden_layers.append(nn.BatchNorm1d(num_features=cfg.n_hidden))
        self.hidden = nn.Sequential(*hidden_layers)
        self.output = nn.Sequential(nn.Linear(cfg.n_hidden, cfg.n_clusters), nn.Softmax(dim=1))

    def forward(self, x):
        hidden = self.hidden(x)
        output = self.output(hidden)
        return output, hidden
