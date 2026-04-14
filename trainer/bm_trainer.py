#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch

from .abs_trainer import Trainer
from torch import nn


def disable_grad(model):
    for name, param in model.named_parameters():
        param.requires_grad = False


def enable_grad(model):
    for name, param in model.named_parameters():
        param.requires_grad = True


class BMTrainer(Trainer):

    ########## Override start ##########

    def __init__(self, model, train_loader, valid_loader, config):
        super().__init__(model, train_loader, valid_loader, config)

    def train_step(self, batch, batch_idx):
        return self.share_step(batch, batch_idx, val=False)

    def valid_step(self, batch, batch_idx):
        return self.share_step(batch, batch_idx, val=True)

    ########## Override end ##########

    def share_step(self, batch, batch_idx, val=False):
        log_type = 'Validation' if val else 'Train'

        if isinstance(self.model, nn.parallel.DistributedDataParallel):
            loss, (loss_vel, loss_den, loss_aux) = self.model.module._train(batch, self.config.temperature)
        else:
            loss, (loss_vel, loss_den, loss_aux) = self.model._train(batch, self.config.temperature)

        self.log(f'Loss/{log_type}', loss, batch_idx, val)
        self.log(f'Vel Loss/{log_type}', loss_vel, batch_idx, val)
        self.log(f'Den Loss/{log_type}', loss_den, batch_idx, val)
        self.log(f'Aux Loss/{log_type}', loss_aux, batch_idx, val)

        return loss
