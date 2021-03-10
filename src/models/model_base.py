import torch.nn as nn


class ModelBase(nn.Module):
    def __init__(self):
        """
        Model base class
        """
        super().__init__()

        self.fusion = None
        self.optimizer = None
        self.loss = None

    def calc_losses(self, ignore_in_total=tuple()):
        return self.loss(self, ignore_in_total=ignore_in_total)

    def train_step(self, batch, epoch, it, n_batches):
        self.optimizer.zero_grad()
        _ = self(batch)
        losses = self.calc_losses()
        losses["tot"].backward()
        self.optimizer.step(epoch + it / n_batches)
        return losses
