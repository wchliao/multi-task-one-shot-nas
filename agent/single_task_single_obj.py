import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
from model import SimpleModel
from controller import Controller
from mask_sampler import MaskSampler
from .base import BaseAgent


class SingleTaskSingleObjectiveAgent(BaseAgent):
    def __init__(self, architecture, search_space, task_info):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.search_size = len(search_space)

        self.model = SimpleModel(architecture=architecture,
                                 search_space=search_space,
                                 in_channels=task_info.num_channels,
                                 num_classes=[task_info.num_classes]
                                 )
        self.submodel = self.model.submodel

        self.controller = Controller(num_outputs=self.model.mask_size)
        self.baseline = None

        self.finalmodel_mask = None
        self.finalmodel = None

        self.mask_sampler = MaskSampler(mask_size=self.model.mask_size, controller=self.controller)
        self.model = nn.DataParallel(self.model).to(self.device)

        self.epoch = {'pretrain': 0, 'controller': 0, 'final': 0}
        self.accuracy = {'pretrain': [], 'controller': [], 'final': []}


    def train(self, train_data, valid_data, test_data, configs, save_model, save_history, path, verbose):

        # Pretrain

        if self.epoch['pretrain'] < configs.pretrain.num_epochs:
            self._pretrain(train_data=train_data,
                           test_data=test_data,
                           configs=configs.pretrain,
                           save_model=save_model,
                           save_history=save_history,
                           path=os.path.join(path, 'pretrain'),
                           verbose=verbose
                           )

        # Train controller

        if self.epoch['controller'] < configs.controller.num_epochs:
            self._train_controller(valid_data=valid_data,
                                   test_data=test_data,
                                   configs=configs.controller,
                                   save_model=save_model,
                                   save_history=save_history,
                                   path=os.path.join(path, 'controller'),
                                   verbose=verbose
                                   )

        # Train final model

        if self.epoch['final'] < configs.final.num_epochs:
            self._finaltrain(train_data=train_data,
                             test_data=test_data,
                             configs=configs.final,
                             save_model=save_model,
                             save_history=save_history,
                             path=os.path.join(path, 'final'),
                             verbose=verbose
                             )


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

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=configs.lr, momentum=configs.momentum, weight_decay=configs.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=configs.lr_decay_epoch, gamma=configs.lr_decay)

        for epoch in range(self.epoch['pretrain']):
            scheduler.step()

        for epoch in range(self.epoch['pretrain'], configs.num_epochs):
            scheduler.step()
            dropout = configs.dropout * epoch / configs.num_epochs

            for inputs, labels in train_data:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                masks = self.mask_sampler.rand().gt(dropout)
                outputs = self.model(inputs, masks)
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

        for epoch in range(self.epoch['controller'], configs.num_epochs):
            masks, log_probs = self.mask_sampler.sample()
            accuracy = self._eval_model(valid_data, masks)

            if self.baseline is None:
                self.baseline = accuracy
            else:
                self.baseline = configs.baseline_decay * self.baseline + (1 - configs.baseline_decay) * accuracy

            advantage = accuracy - self.baseline
            loss = -log_probs * advantage
            loss = loss.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if verbose or save_history:
                masks, _ = self.mask_sampler.sample(sample_best=True, grad=False)
                self.accuracy['controller'].append(self._eval_model(test_data, masks))

            if verbose:
                print('[Controller][Epoch {}] Accuracy: {}'.format(epoch + 1, self.accuracy['controller'][-1]))

            if epoch % configs.save_epoch == 0 and save_model:
                self._save_controller(path)
                self.epoch['controller'] = epoch + 1
                self._save_epoch('controller', path)

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
            self.finalmodel_mask, _ = self.mask_sampler.sample(sample_best=True, grad=False)
            self.finalmodel_mask = self.finalmodel_mask[0]
            self.finalmodel = self.submodel(self.finalmodel_mask)
            self.finalmodel = nn.DataParallel(self.finalmodel).to(self.device)

        self.finalmodel.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.finalmodel.parameters(), lr=configs.lr, momentum=configs.momentum, weight_decay=configs.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=configs.lr_decay_epoch, gamma=configs.lr_decay)

        for epoch in range(self.epoch['final']):
            scheduler.step()

        for epoch in range(self.epoch['final'], configs.num_epochs):
            scheduler.step()

            for inputs, labels in train_data:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.finalmodel(inputs)
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


    def eval(self, data):
        if self.finalmodel is None:
            return self._eval_model(data)
        else:
            return self._eval_final(data)


    def _eval_model(self, data, masks=None):
        if masks is None:
            masks, _ = self.mask_sampler.sample(sample_best=True, grad=False)

        model = lambda x: self.model(x, masks)
        accuracy = self._eval(data, model)

        return accuracy


    def _eval_final(self, data):
        self.finalmodel.eval()
        accuracy = self._eval(data, self.finalmodel)
        self.finalmodel.train()

        return accuracy


    def _eval(self, data, model):
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


    def save(self, path='saved_models/default/'):
        self._save_pretrain(os.path.join(path, 'pretrain'))
        self._save_controller(os.path.join(path, 'controller'))
        self._save_final(os.path.join(path, 'final'))

        for key in ['pretrain', 'controller', 'final']:
            self._save_epoch(key, os.path.join(path, key))
            self._save_accuracy(key, os.path.join(path, key))


    def _save_pretrain(self, path='saved_models/default/pretrain/'):
        if not os.path.isdir(path):
            os.makedirs(path)
        torch.save(self.model.state_dict(), os.path.join(path, 'model'))


    def _save_controller(self, path='saved_models/default/controller/'):
        if not os.path.isdir(path):
            os.makedirs(path)
        torch.save(self.controller.state_dict(), os.path.join(path, 'model'))


    def _save_final(self, path='saved_models/default/final/'):
        if not os.path.isdir(path):
            os.makedirs(path)

        with open(os.path.join(path, 'masks'), 'w') as f:
            json.dump(self.finalmodel_mask.tolist(), f)

        torch.save(self.finalmodel.state_dict(), os.path.join(path, 'model'))


    def _save_epoch(self, key, path='saved_models/default/'):
        if not os.path.isdir(path):
            os.makedirs(path)

        with open(os.path.join(path, 'last_epoch.json'), 'w') as f:
            json.dump(self.epoch[key], f)


    def _save_accuracy(self, key, path='saved_models/default/'):
        if not os.path.isdir(path):
            os.makedirs(path)
        filename = os.path.join(path, 'history.json')

        with open(filename, 'w') as f:
            json.dump(self.accuracy[key], f)


    def load(self, path='saved_models/default/'):
        self._load_pretrain(os.path.join(path, 'pretrain'))
        self._load_controller(os.path.join(path, 'controller'))
        self._load_final(os.path.join(path, 'final'))

        for key in ['pretrain', 'controller', 'final']:
            self._load_epoch(key, os.path.join(path, key))
            self._load_accuracy(key, os.path.join(path, key))


    def _load_pretrain(self, path='saved_models/default/pretrain/'):
        try:
            filename = os.path.join(path, 'model')
            self.model.load_state_dict(torch.load(filename))

        except FileNotFoundError:
            pass


    def _load_controller(self, path='saved_models/default/controller/'):
        try:
            filename = os.path.join(path, 'model')
            self.controller.load_state_dict(torch.load(filename))

        except FileNotFoundError:
            pass


    def _load_final(self, path='saved_models/default/final/'):
        try:
            with open(os.path.join(path, 'masks'), 'r') as f:
                self.finalmodel_mask = json.load(f)
            self.finalmodel_mask = torch.tensor(self.finalmodel_mask)
            self.finalmodel = self.submodel(self.finalmodel_mask)
            self.finalmodel = nn.DataParallel(self.finalmodel).to(self.device)

            filename = os.path.join(path, 'model')
            self.finalmodel.load_state_dict(torch.load(filename))

        except FileNotFoundError:
            pass


    def _load_epoch(self, key, path='saved_models/default/'):
        try:
            filename = os.path.join(path, 'last_epoch.json')
            with open(filename, 'r') as f:
                self.epoch[key] = json.load(f)

        except FileNotFoundError:
            self.epoch[key] = 0


    def _load_accuracy(self, key, path='saved_models/default/'):
        try:
            with open(os.path.join(path, 'history.json'), 'r') as f:
                self.accuracy[key] = json.load(f)

        except FileNotFoundError:
            pass
