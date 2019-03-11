import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import numpy as np
from model import SimpleModel
from controller import Controller
from utils import MaskSampler, MultiModelSize
from .multi_task_single_obj_single_model import MultiTaskSingleObjectiveSingleModelAgent


class MultiTaskSingleObjectiveMultiModelAgent(MultiTaskSingleObjectiveSingleModelAgent):
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

        self.controller = Controller(num_outputs=self.model.mask_size, num_tasks=self.num_tasks)
        self.baseline = [None for _ in range(self.num_tasks)]

        self.finalmodel_mask = None
        self.finalmodel = None

        self.mask_sampler = MaskSampler(mask_size=self.model.mask_size, controller=self.controller)
        self.compute_model_size = MultiModelSize(architecture, search_space, task_info.num_channels, task_info.num_classes)

        self.model = nn.DataParallel(self.model).to(self.device)

        self.epoch = {'pretrain': 0, 'controller': 0, 'final': 0}
        self.accuracy = {'pretrain': [], 'controller': [], 'final': []}


    def _train_controller(self,
                          valid_data,
                          test_data,
                          configs,
                          save_model=False,
                          save_history=False,
                          path='saved_models/default/controller/',
                          verbose=False
                          ):

        self.controller.train()

        optimizer = optim.Adam(self.controller.parameters(), lr=configs.lr)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=configs.lr_decay_epoch, gamma=configs.lr_decay)

        for epoch in range(self.epoch['controller'], configs.num_epochs):
            scheduler.step()
            for t in range(self.num_tasks):
                masks, log_probs = self.mask_sampler.sample(task=t)
                accuracy = self._eval_model(valid_data, masks, t)

                if self.baseline[t] is None:
                    self.baseline[t] = accuracy
                else:
                    self.baseline[t] = configs.baseline_decay * self.baseline[t] + (1 - configs.baseline_decay) * accuracy

                advantage = accuracy - self.baseline[t]
                loss = -log_probs * advantage
                loss = loss.sum()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if verbose or save_history:
                accuracy = []
                for t in range(self.num_tasks):
                    masks, _ = self.mask_sampler.sample(task=t, sample_best=True, grad=False)
                    accuracy.append(self._eval_model(test_data, masks, t))
                self.accuracy['controller'].append(np.mean(accuracy))

            if verbose:
                print('[Controller][Epoch {}] Accuracy: {}'.format(epoch + 1, self.accuracy['controller'][-1]))

            if epoch % configs.save_epoch == 0:
                if save_model:
                    self._save_controller(path)
                    self.epoch['controller'] = epoch + 1
                    self._save_epoch('controller', path)

                if save_history:
                    self._save_accuracy('controller', path)

        if save_model:
            self._save_controller(path)
            self.epoch['controller'] = configs.num_epochs
            self._save_epoch('controller', path)

        if save_history:
            self._save_accuracy('controller', path)


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
            self.finalmodel_mask = []
            self.finalmodel = []
            for t in range(self.num_tasks):
                masks, _ = self.mask_sampler.sample(task=t, sample_best=True, grad=False, batch=False)
                model = self.submodel(masks, t)
                model = nn.DataParallel(model).to(self.device)
                self.finalmodel_mask.append(masks)
                self.finalmodel.append(model)

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

            if epoch % configs.save_epoch == 0:
                if save_model:
                    self._save_final(path)
                    self.epoch['final'] = epoch + 1
                    self._save_epoch('final', path)

                if save_history:
                    self._save_accuracy('final', path)

        if save_model:
            self._save_final(path)
            self.epoch['final'] = configs.num_epochs
            self._save_epoch('final', path)

        if save_history:
            self._save_accuracy('final', path)


    def _eval_model(self, data, masks=None, task=None):
        if masks is None:
            masks, _ = self.mask_sampler.sample(sample_best=True, grad=False)

        if task is None:
            model = lambda x, t: self.model(x, masks, t)
            accuracy = self._eval(data, model)
        else:
            model = lambda x: self.model(x, masks, task)
            accuracy = self._eval_single_task(data.get_loader(task), model)

        return accuracy


    def _eval_final(self, data, task=None):
        for model in self.finalmodel:
            model.eval()

        if task is None:
            model = lambda x, t: self.finalmodel[t](x)
            accuracy = self._eval(data, model)
        else:
            model = lambda x: self.finalmodel[task](x)
            accuracy = self._eval_single_task(data.get_loader(task), model)

        for model in self.finalmodel:
            model.train()

        return accuracy


    def _eval_single_task(self, data, model):
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in data:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                _, predict_labels = torch.max(outputs.detach(), 1)

                total += labels.size(0)
                correct += (predict_labels == labels).sum().item()

            return correct / total


    def _eval(self, data, model):
        accuracy = []

        for t in range(self.num_tasks):
            model_t = lambda x: model(x, t)
            acc = self._eval_single_task(data.get_loader(t), model_t)
            accuracy.append(acc)

        return np.mean(accuracy)


    def _save_final(self, path='saved_models/default/final/'):
        if not os.path.isdir(path):
            os.makedirs(path)

        with open(os.path.join(path, 'masks'), 'w') as f:
            json.dump([mask.tolist() for mask in self.finalmodel_mask], f)

        for t, model in enumerate(self.finalmodel):
            torch.save(model.state_dict(), os.path.join(path, 'model{}'.format(t)))


    def _load_final(self, path='saved_models/default/final/'):
        try:
            with open(os.path.join(path, 'masks'), 'r') as f:
                self.finalmodel_mask = json.load(f)
            self.finalmodel_mask = torch.tensor(self.finalmodel_mask)
            self.finalmodel = [self.submodel(masks, t) for t, masks in enumerate(self.finalmodel_mask)]
            self.finalmodel = [nn.DataParallel(model).to(self.device) for model in self.finalmodel]

            for t, model in enumerate(self.finalmodel):
                filename = os.path.join(path, 'model{}'.format(t))
                model.load_state_dict(torch.load(filename))

        except FileNotFoundError:
            pass
