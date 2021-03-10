import helpers
from lib.loss import Loss
from lib.fusion import get_fusion_module
from lib.optimizer import Optimizer
from lib.backbones import Backbones
from models.model_base import ModelBase
from models.clustering_module import DDC


class SiMVC(ModelBase):
    def __init__(self, cfg):
        """
        Implementation of the SiMVC model.

        :param cfg: Model config. See `config.defaults.SiMVC` for documentation on the config object.
        """
        super().__init__()

        self.cfg = cfg
        self.output = self.hidden = self.fused = self.backbone_outputs = None

        # Define Backbones and Fusion modules
        self.backbones = Backbones(cfg.backbone_configs)
        self.fusion = get_fusion_module(cfg.fusion_config, self.backbones.output_sizes)
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
        self.output, self.hidden = self.ddc(self.fused)
        return self.output
