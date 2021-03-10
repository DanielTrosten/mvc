import torch as th
import torch.nn as nn

import helpers
from lib.loss import Loss
from lib.optimizer import Optimizer
from lib.backbones import Backbones, MLP
from lib.fusion import get_fusion_module
from models.clustering_module import DDC
from models.model_base import ModelBase


class CoMVC(ModelBase):
    def __init__(self, cfg):
        """
        Implementation of the CoMVC model.

        :param cfg: Model config. See `config.defaults.CoMVC` for documentation on the config object.
        """
        super().__init__()

        self.cfg = cfg
        self.output = self.hidden = self.fused = self.backbone_outputs = self.projections = None

        # Define Backbones and Fusion modules
        self.backbones = Backbones(cfg.backbone_configs)
        self.fusion = get_fusion_module(cfg.fusion_config, self.backbones.output_sizes)

        bb_sizes = self.backbones.output_sizes
        assert all([bb_sizes[0] == s for s in bb_sizes]), f"CoMVC requires all backbones to have the same " \
                                                          f"output size. Got: {bb_sizes}"

        if cfg.projector_config is None:
            self.projector = nn.Identity()
        else:
            self.projector = MLP(cfg.projector_config, input_size=bb_sizes[0])

        # Define clustering module
        self.ddc = DDC(input_dim=self.fusion.output_size, cfg=cfg.cm_config)
        # Define loss-module
        self.loss = Loss(cfg=cfg.loss_config)
        # Initialize weights.
        self.apply(helpers.he_init_weights)
        # Instantiate optimizer
        self.optimizer = Optimizer(cfg.optimizer_config, self.parameters())

    def forward(self, views):
        self.backbone_outputs = self.backbones(views)
        self.fused = self.fusion(self.backbone_outputs)
        self.projections = self.projector(th.cat(self.backbone_outputs, dim=0))
        self.output, self.hidden = self.ddc(self.fused)
        return self.output

