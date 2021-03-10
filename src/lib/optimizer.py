import torch as th


class Optimizer:
    def __init__(self, cfg, params):
        """
        Wrapper class for optimizers

        :param cfg: Optimizer config
        :type cfg: config.defaults.Optimizer
        :param params: Parameters to associate with the optimizer
        :type params:
        """
        self.clip_norm = cfg.clip_norm
        self.params = params
        self._opt = th.optim.Adam(params, lr=cfg.learning_rate)
        if cfg.scheduler_step_size is not None:
            assert cfg.scheduler_gamma is not None
            self._sch = th.optim.lr_scheduler.StepLR(self._opt, step_size=cfg.scheduler_step_size,
                                                     gamma=cfg.scheduler_gamma)
        else:
            self._sch = None

    def zero_grad(self):
        return self._opt.zero_grad()

    def step(self, epoch):
        if self._sch is not None:
            # Only step the scheduler at integer epochs, and don't step on the first epoch.
            if epoch.is_integer() and epoch > 0:
                self._sch.step()

        if self.clip_norm is not None:
            th.nn.utils.clip_grad_norm_(self.params, self.clip_norm)

        out = self._opt.step()
        return out
