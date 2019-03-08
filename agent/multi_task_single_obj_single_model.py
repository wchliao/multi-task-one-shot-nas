import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import numpy as np
from model import SimpleModel
from controller import Controller
from utils import MaskSampler, SingleModelSize
from .single_task_single_obj import SingleTaskSingleObjectiveAgent


class MultiTaskSingleObjectiveSingleModelAgent(SingleTaskSingleObjectiveAgent):
    def __init__(self, architecture, search_space, task_info):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.search_size = len(search_space)
        self.num_tasks = task_info.num_tasks

        self.model = SimpleModel(architecture=architecture,
                                 search_space=search_space,
                                 in_channels=task_info.num_channels,
                                 num_classes=task_info.num_classes
                                 )
        self.submodel = self.model.submodel

        self.controller = Controller(num_outputs=self.model.mask_size)
        self.baseline = None

        self.finalmodel_mask = None
        self.finalmodel = None

        self.mask_sampler = MaskSampler(mask_size=self.model.mask_size, controller=self.controller)
        self.compute_model_size = SingleModelSize(architecture, search_space, task_info.num_channels, sum(task_info.num_classes))

        self.model = nn.DataParallel(self.model).to(self.device)

        self.epoch = {'pretrain': 0, 'controller': 0, 'final': 0}
        self.accuracy = {'pretrain': [], 'controller': [], 'final': []}


    def _pretrain(self,
                  train_data,
                  test_data,
                  configs,
                  save_model=False,
                  save_history=False,
                  path='saved_models/default/pretrain/',
                  verbose=False
                  ):

        self.model.train()

        dataloader = train_data.get_loader()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=configs.lr, momentum=configs.momentum, weight_decay=configs.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=configs.lr_decay_epoch, gamma=configs.lr_decay)

        for epoch in range(self.epoch['pretrain']):
            scheduler.step()

        for epoch in range(self.epoch['pretrain'], configs.num_epochs):
            scheduler.step()
            dropout = configs.dropout * epoch / configs.num_epochs

            for inputs, labels, task in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                masks = self.mask_sampler.rand().gt(dropout)
                outputs = self.model(inputs, masks, task)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if verbose or save_history:
                masks = self.mask_sampler.ones()
                self.accuracy['pretrain'].append(self._eval_model(test_data, masks))

            if verbose:
                print('[Pretrain][Epoch {}] Accuracy: {}'.format(epoch + 1, self.accuracy['pretrain'][-1]))

            if epoch % configs.save_epoch == 0 and save_model:
                self._save_pretrain(path)
                self.epoch['pretrain'] = epoch + 1
                self._save_epoch('pretrain', path)

        if save_model:
            self._save_pretrain(path)
            self.epoch['pretrain'] = configs.num_epochs
            self._save_epoch('pretrain', path)


    def _finaltrain(self,
                    train_data,
                    test_data,
                    configs,
                    save_model=False,
                    save_history=False,
                    path='saved_models/default/final/',
                    verbose=False
                    ):

        if self.finalmodel is None:
            self.finalmodel_mask, _ = self.mask_sampler.sample(sample_best=True, grad=False, batch=False)
            self.finalmodel = [self.submodel(self.finalmodel_mask, task) for task in range(self.num_tasks)]
            self.finalmodel = [nn.DataParallel(m).to(self.device) for m in self.finalmodel]

        for model in self.finalmodel:
            model.train()

        dataloader = train_data.get_loader()
        criterion = nn.CrossEntropyLoss()
        optimizers = [optim.SGD(model.parameters(), lr=configs.lr, momentum=configs.momentum, weight_decay=configs.weight_decay) for model in self.finalmodel]
        schedulers = [optim.lr_scheduler.MultiStepLR(optimizer, milestones=configs.lr_decay_epoch, gamma=configs.lr_decay) for optimizer in optimizers]

        for epoch in range(self.epoch['final']):
            for scheduler in schedulers:
                scheduler.step()

        for epoch in range(self.epoch['final'], configs.num_epochs):
            for scheduler in schedulers:
                scheduler.step()

            for inputs, labels, task in dataloader:
                model = self.finalmodel[task]
                optimizer = optimizers[task]

                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if verbose or save_history:
                self.accuracy['final'].append(self._eval_final(test_data))

            if verbose:
                print('[Final][Epoch {}] Accuracy: {}'.format(epoch + 1, self.accuracy['final'][-1]))

            if epoch % configs.save_epoch == 0 and save_model:
                self._save_final(path)
                self.epoch['final'] = epoch + 1
                self._save_epoch('final', path)

        if save_model:
            self._save_final(path)
            self.epoch['final'] = configs.num_epochs
            self._save_epoch('final', path)

        if save_history:
            self._save_accuracy('final', path)


    def _eval_model(self, data, masks=None):
        if masks is None:
            masks, _ = self.mask_sampler.sample(sample_best=True, grad=False)

        model = lambda x, t: self.model(x, masks, t)
        accuracy = self._eval(data, model)

        return accuracy


    def _eval_final(self, data):
        for model in self.finalmodel:
            model.eval()

        model = lambda x, t: self.finalmodel[t](x)
        accuracy = self._eval(data, model)

        for model in self.finalmodel:
            model.train()

        return accuracy


    def _eval(self, data, model):
        correct = [0 for _ in range(self.num_tasks)]
        total = [0 for _ in range(self.num_tasks)]

        with torch.no_grad():
            for t in range(self.num_tasks):
                for inputs, labels in data.get_loader(t):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs, t)
                    _, predict_labels = torch.max(outputs.detach(), 1)

                    total[t] += labels.size(0)
                    correct[t] += (predict_labels == labels).sum().item()

            return np.mean([c / t for c, t in zip(correct, total)])


    def _save_final(self, path='saved_models/default/final/'):
        if not os.path.isdir(path):
            os.makedirs(path)

        with open(os.path.join(path, 'masks'), 'w') as f:
            json.dump(self.finalmodel_mask.tolist(), f)

        for t, model in enumerate(self.finalmodel):
            torch.save(model.state_dict(), os.path.join(path, 'model{}'.format(t)))


    def _load_final(self, path='saved_models/default/final/'):
        try:
            with open(os.path.join(path, 'masks'), 'r') as f:
                self.finalmodel_mask = json.load(f)
            self.finalmodel_mask = torch.tensor(self.finalmodel_mask)

            self.finalmodel = [self.submodel(self.finalmodel_mask, task) for task in range(self.num_tasks)]
            self.finalmodel = [nn.DataParallel(m).to(self.device) for m in self.finalmodel]

            for t, model in enumerate(self.finalmodel):
                filename = os.path.join(path, 'model{}'.format(t))
                model.load_state_dict(torch.load(filename))

        except FileNotFoundError:
            pass