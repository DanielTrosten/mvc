import numpy as np

import helpers
from lib.loss import Loss
from lib.optimizer import Optimizer
from lib.backbones import Backbones
from models.model_base import ModelBase
from models.clustering_module import DDC


class DDCModel(ModelBase):
    def __init__(self, cfg):
        """
        Full DDC model

        :param cfg: DDC model config
        :type cfg: config.defaults.DDCModel
        """
        super().__init__()

        self.cfg = cfg
        self.backbone_output = self.output = self.hidden = None
        self.backbone = Backbones.create_backbone(cfg.backbone_config)
        self.ddc_input_size = np.prod(self.backbone.output_size)
        self.ddc = DDC([self.ddc_input_size], cfg.cm_config)
        self.loss = Loss(cfg.loss_config)

        # Initialize weights.
        self.apply(helpers.he_init_weights)
        # Instantiate optimizer
        self.optimizer = Optimizer(cfg.optimizer_config, self.parameters())

    def forward(self, x):
        if isinstance(x, list):
            # We might get a one-element list as input due to multi-view compatibility.
            assert len(x) == 1
            x = x[0]

        self.backbone_output = self.backbone(x).view(-1, self.ddc_input_size)
        self.output, self.hidden = self.ddc(self.backbone_output)
        return self.output
