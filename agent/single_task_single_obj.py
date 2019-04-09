import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import numpy as np
from model import SimpleModel
from utils import MaskSampler, masks2str, ModelSize
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
        self.compute_model_size = ModelSize(architecture, search_space, task_info.num_channels, task_info.num_classes, batchnorm=True)

        self._init()


    def _init(self):
        self.submodel = self.model.submodel
        self.mask_sampler = MaskSampler(mask_size=self.model.mask_size)
        self.model = nn.DataParallel(self.model).to(self.device)

        # Record

        self.epoch = {'pretrain': 0, 'search': 0, 'final': 0}
        self.accuracy = {'pretrain': [], 'search': [], 'final': []}

        # Search

        self.accuracy_dict_valid = {}
        self.accuracy_dict_test = {}
        self.queue = []
        self.queue_acc = []

        # Final

        self.finalmodel_mask = None
        self.finalmodel = None



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

        # Select final model

        if self.epoch['search'] < configs.search.num_epochs:
            self._search(valid_data=valid_data,
                         test_data=test_data,
                         configs=configs.search,
                         save_model=save_model,
                         save_history=save_history,
                         path=os.path.join(path, 'search'),
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
                masks = self.mask_sampler.rand(dropout=dropout)
                outputs = self.model(inputs, self.mask_sampler.make_batch(masks))
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


    def _search(self,
                valid_data,
                test_data,
                configs,
                save_model=False,
                save_history=False,
                path='saved_models/default/search/',
                verbose=False
                ):

        # Initalization

        if self.epoch['search'] == 0:
            self.queue = [self.mask_sampler.rand(dropout=i/(configs.num_samples-1)) for i in range(configs.num_samples)]
            self.queue_acc = []

            for masks in self.queue:
                masks_str = masks2str(masks)
                accuracy = self._eval_model(valid_data, masks)
                self.accuracy_dict_valid[masks_str] = accuracy
                self.queue_acc.append(accuracy)

        # Search

        for epoch in range(self.epoch['search'], configs.num_epochs):
            generated = []
            generated_acc = []

            for old_masks in self.queue:
                new_masks = self.mask_sampler.mutate(old_masks, configs.mutate_prob)
                new_masks_str = masks2str(new_masks)

                if new_masks_str not in self.accuracy_dict_valid:
                    self.accuracy_dict_valid[new_masks_str] = self._eval_model(valid_data, new_masks)

                generated.append(new_masks)
                generated_acc.append(self.accuracy_dict_valid[new_masks_str])

            candidates = self.queue + generated
            candidates_acc = self.queue_acc + generated_acc
            order = np.argsort(candidates_acc)[::-1][:configs.num_samples]
            self.queue = [candidates[i] for i in order]
            self.queue_acc = [candidates_acc[i] for i in order]
            best_masks = self.queue[0]
            best_masks_str = masks2str(best_masks)

            if verbose or save_history:
                if best_masks_str not in self.accuracy_dict_test:
                    self.accuracy_dict_test[best_masks_str] = self._eval_model(test_data, best_masks)
                self.accuracy['search'].append(self.accuracy_dict_test[best_masks_str])

            if verbose:
                print('[Search][Epoch {}] Accuracy: {}'.format(epoch + 1, self.accuracy['search'][-1]))

            if epoch % configs.save_epoch == 0:
                if save_model:
                    self._save_search(path)
                    self.epoch['search'] = epoch + 1
                    self._save_epoch('search', path)

                if save_history:
                    self._save_accuracy('search', path)

        if save_model:
            self._save_search(path)
            self.epoch['search'] = configs.num_epochs
            self._save_epoch('search', path)

        if save_history:
            self._save_accuracy('search', path)


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
            self.finalmodel_mask = self.queue[0]
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


    def eval(self, data):
        accuracy = self._eval_final(data)
        model_size = self.compute_model_size.compute(self.finalmodel_mask)

        return accuracy, model_size


    def _eval_model(self, data, masks):
        masks = self.mask_sampler.make_batch(masks)
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


    def _save_pretrain(self, path='saved_models/default/pretrain/'):
        if not os.path.isdir(path):
            os.makedirs(path)
        torch.save(self.model.state_dict(), os.path.join(path, 'model'))


    def _save_search(self, path='saved_models/default/search/'):
        if not os.path.isdir(path):
            os.makedirs(path)

        with open(os.path.join(path, 'accuracy_dict_valid.json'), 'w') as f:
            json.dump(self.accuracy_dict_valid, f)
        with open(os.path.join(path, 'accuracy_dict_test.json'), 'w') as f:
            json.dump(self.accuracy_dict_test, f)
        with open(os.path.join(path, 'queue.json'), 'w') as f:
            json.dump([masks.tolist() for masks in self.queue], f)
        with open(os.path.join(path, 'queue_acc.json'), 'w') as f:
            json.dump(self.queue_acc, f)


    def _save_final(self, path='saved_models/default/final/'):
        if not os.path.isdir(path):
            os.makedirs(path)

        with open(os.path.join(path, 'masks.json'), 'w') as f:
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
        self._load_search(os.path.join(path, 'search'))
        self._load_final(os.path.join(path, 'final'))

        for key in ['pretrain', 'search', 'final']:
            self._load_epoch(key, os.path.join(path, key))
            self._load_accuracy(key, os.path.join(path, key))


    def _load_pretrain(self, path='saved_models/default/pretrain/'):
        try:
            filename = os.path.join(path, 'model')
            self.model.load_state_dict(torch.load(filename))

        except FileNotFoundError:
            pass


    def _load_search(self, path='saved_models/default/search/'):
        try:
            with open(os.path.join(path, 'accuracy_dict_valid.json')) as f:
                self.accuracy_dict_valid = json.load(f)
            with open(os.path.join(path, 'accuracy_dict_test.json')) as f:
                self.accuracy_dict_test = json.load(f)
            with open(os.path.join(path, 'queue.json')) as f:
                self.queue = json.load(f)
                self.queue = [torch.tensor(masks, dtype=torch.uint8) for masks in self.queue]
            with open(os.path.join(path, 'queue_acc.json')) as f:
                self.queue_acc = json.load(f)

        except FileNotFoundError:
            self.queue = []
            self.queue_acc = []


    def _load_final(self, path='saved_models/default/final/'):
        try:
            with open(os.path.join(path, 'masks.json'), 'r') as f:
                self.finalmodel_mask = json.load(f)
            self.finalmodel_mask = torch.tensor(self.finalmodel_mask, dtype=torch.uint8)
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
