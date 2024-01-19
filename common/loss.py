from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F



class Mix(nn.Module):
    def __init__(self, losses, coefficients: Optional[dict]=None):
        super(Mix, self).__init__()
        self.losses = losses
        self.coefficients = coefficients

        if self.coefficients is None:
            self.coefficients = { k:1 for k in self.losses }

    def forward(self, target, predict):

        losses_results = {k:self.losses[k](target, predict) for k in self.losses }

        assert self.coefficients is not None
        loss = sum([
            losses_results[k]*self.coefficients[k]
            for k in losses_results if losses_results[k] is not None
        ]) / (len(losses_results))

        return loss, losses_results

    @staticmethod
    def normalize_data(data):
        return (data - torch.min(data)) / (torch.max(data) - torch.min(data))


class BCE_Lossv2(nn.Module):
    def __init__(
        self,
        output_key: Union[int, str]=0,
        target_key: Union[int, str]=0,
        bg_weight=1,
    ):
        super(BCE_Lossv2, self).__init__()
        self.output_key = output_key
        self.target_key = target_key
        self.bg_weight=bg_weight

    def forward(self, target, predict):

        assert (target[self.target_key].shape == predict[self.output_key].shape)

        pred = predict[self.output_key].view(-1)
        gt = target[self.target_key].view(-1)

        loss = F.binary_cross_entropy(pred, gt, reduction='mean')

        return loss


class Dice_loss_jointv2(nn.Module):
    def __init__(
        self,
        output_key: Union[int, str]=0,
        target_key: Union[int, str]=0,
        force_binary: bool=False,
        threshold: float=0.5,
    ):
        super(Dice_loss_jointv2, self).__init__()
        self.output_key = output_key
        self.target_key = target_key
        self.force_binary = force_binary
        self.threshold = threshold

    def forward(self, target, predict):

        assert (target[self.target_key].shape == predict[self.output_key].shape), \
            f'{target[self.target_key].shape} != {predict[self.output_key].shape}'
        shape = target[self.target_key].shape

        pred = predict[self.output_key].view(shape[0], shape[1], -1)
        gt = target[self.target_key].view(shape[0], shape[1], -1)

        if self.force_binary:
            # pred = (pred > self.threshold).float()
            gt = (gt > self.threshold).float()

        intersection = (pred*gt).sum(dim=(0,2)) + 1e-6
        union = (pred**2 + gt).sum(dim=(0,2)) + 2e-6
        dice = 2.0*intersection / union

        return (1.0 - torch.mean(dice))
