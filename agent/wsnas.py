import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import numpy as np
import copy
from model import ComplexModel
from utils import MaskSampler, ComplexModelSize, pareto_front, pareto_front_full
from .base import BaseMultiObjectiveAgent


class WSNASAgent(BaseMultiObjectiveAgent):
    def __init__(self, architecture, search_space, task_info):
        super(WSNASAgent, self).__init__(architecture, search_space, task_info)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.search_size = len(search_space)
        self.num_tasks = task_info.num_tasks

        self.model = ComplexModel(architecture=architecture,
                                  search_space=search_space,
                                  in_channels=task_info.num_channels,
                                  num_classes=task_info.num_classes
                                  )
        self.compute_model_size = ComplexModelSize(architecture, search_space, task_info.num_channels, task_info.num_classes)

        self.submodel = self.model.submodel
        self.mask_sampler = MaskSampler(mask_size=self.model.mask_size)
        self.model = nn.DataParallel(self.model).to(self.device)

        # Record

        self.epoch = {'pretrain': 0, 'search': 0, 'final': 0}
        self.accuracy = {'pretrain': [], 'final': []}

        # Search

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

        # Search final models

        if self.epoch['search'] < configs.search.num_epochs:
            self._search(valid_data=valid_data,
                         configs=configs.search,
                         save_model=save_model,
                         path=os.path.join(path, 'search'),
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
                shared_masks = self.mask_sampler.rand(dropout=dropout)
                task_masks = self.mask_sampler.rand(dropout=dropout)
                outputs = self.model(inputs, self.mask_sampler.make_batch(shared_masks), self.mask_sampler.make_batch(task_masks), task)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if verbose or save_history:
                shared_masks = self.mask_sampler.ones()
                tasks_masks = [self.mask_sampler.ones()] * self.num_tasks
                self.accuracy['pretrain'].append(self._eval_model(test_data, shared_masks, tasks_masks))

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
                configs,
                save_model=False,
                path='saved_models/default/search/',
                verbose=False
                ):

        # Initalization

        if self.epoch['search'] == 0:
            num_samples_root = int(np.ceil(configs.num_samples**0.5))
            shared_queue = [self.mask_sampler.rand(dropout=i/(num_samples_root-1)) for i in range(num_samples_root)]
            tasks_queue = [[self.mask_sampler.rand(dropout=i/(num_samples_root-1)) for _ in range(self.num_tasks)] for i in range(num_samples_root)]
            self.queue = [(i, j) for i in shared_queue for j in tasks_queue]
            self.queue_obj = []

            for shared_masks, tasks_masks in self.queue:
                accuracy = self._eval_model(valid_data, shared_masks, tasks_masks)
                model_size = self.compute_model_size.compute(shared_masks, tasks_masks)
                self.queue_obj.append((accuracy, -model_size))

        # Search

        for epoch in range(self.epoch['search'], configs.num_epochs):
            generated = []
            generated_obj = []

            for old_masks in self.queue:
                if epoch % 2 == 0:
                    shared_masks = self.mask_sampler.mutate(old_masks[0], configs.mutate_prob)
                    tasks_masks = copy.deepcopy(old_masks[1])
                else:
                    shared_masks = copy.deepcopy(old_masks[0])
                    tasks_masks = [self.mask_sampler.mutate(m, configs.mutate_prob) for m in old_masks[1]]

                accuracy = self._eval_model(valid_data, shared_masks, tasks_masks)
                model_size = self.compute_model_size.compute(shared_masks, tasks_masks)

                generated.append((shared_masks, tasks_masks))
                generated_obj.append((accuracy, -model_size))

            candidates = self.queue + generated
            candidates_obj = self.queue_obj + generated_obj
            self.queue_obj, order = pareto_front_full(candidates_obj, num=configs.num_samples)
            self.queue = [candidates[i] for i in order]

            if verbose:
                print('[Search][Epoch {}]'.format(epoch + 1))

            if epoch % configs.save_epoch == 0 and save_model:
                self._find_pareto_front()
                self._save_search(path)
                self.epoch['search'] = epoch + 1
                self._save_epoch('search', path)

        if save_model:
            self._find_pareto_front()
            self._save_search(path)
            self.epoch['search'] = configs.num_epochs
            self._save_epoch('search', path)


    def _find_pareto_front(self):
        self.pareto_front_obj, order = pareto_front(self.queue_obj)
        self.pareto_front = [self.queue[i] for i in order]
        order = np.argsort([obj[0] for obj in self.pareto_front_obj])[::-1]
        self.pareto_front = [self.pareto_front[i] for i in order]
        self.pareto_front_obj = [self.pareto_front_obj[i] for i in order]


    def finaltrain(self,
                   train_data,
                   test_data,
                   configs,
                   save_model=False,
                   save_history=False,
                   path='saved_models/default/final/',
                   id=None,
                   verbose=False
                   ):

        path = os.path.join(path, str(id))

        if self.finalmodel is None:
            self.finalmodel_mask = self.pareto_front[id]
            self.finalmodel = [self.submodel(self.finalmodel_mask[0], self.finalmodel_mask[1][task], task) for task in range(self.num_tasks)]
            self.finalmodel = [nn.DataParallel(m).to(self.device) for m in self.finalmodel]

        if save_model:
            self._save_model_size(path)

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


    def eval(self, data):
        accuracy = self._eval_final(data)
        model_size = self.compute_model_size.compute(self.finalmodel_mask[0], self.finalmodel_mask[1])

        return accuracy, model_size


    def _eval_model(self, data, shared_masks, tasks_masks):
        shared_masks = self.mask_sampler.make_batch(shared_masks)
        tasks_masks = [self.mask_sampler.make_batch(m) for m in tasks_masks]
        model = lambda x, t: self.model(x, shared_masks, tasks_masks[t], t)
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


    def _save_pretrain(self, path='saved_models/default/pretrain/'):
        if not os.path.isdir(path):
            os.makedirs(path)
        torch.save(self.model.state_dict(), os.path.join(path, 'model'))


    def _save_search(self, path='saved_models/default/search/'):
        if not os.path.isdir(path):
            os.makedirs(path)

        with open(os.path.join(path, 'queue.json'), 'w') as f:
            json.dump([[masks[0].tolist(), [m.tolist() for m in masks[1]]] for masks in self.queue], f)
        with open(os.path.join(path, 'queue_obj.json'), 'w') as f:
            json.dump(self.queue_obj, f)
        with open(os.path.join(path, 'pareto_front.json'), 'w') as f:
            json.dump([[masks[0].tolist(), [m.tolist() for m in masks[1]]] for masks in self.pareto_front], f)
        with open(os.path.join(path, 'pareto_front_obj.json'), 'w') as f:
            json.dump(self.pareto_front_obj, f)


    def _save_final(self, path='saved_models/default/final/'):
        if not os.path.isdir(path):
            os.makedirs(path)

        with open(os.path.join(path, 'masks.json'), 'w') as f:
            json.dump([self.finalmodel_mask[0].tolist(), [m.tolist() for m in self.finalmodel_mask[1]]], f)

        for t, model in enumerate(self.finalmodel):
            torch.save(model.state_dict(), os.path.join(path, 'model{}'.format(t)))


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


    def _save_model_size(self, path='saved_models/default/'):
        if not os.path.isdir(path):
            os.makedirs(path)
        filename = os.path.join(path, 'model_size.json')

        with open(filename, 'w') as f:
            json.dump(self.compute_model_size.compute(self.finalmodel_mask[0], self.finalmodel_mask[1]), f)


    def load(self, path='saved_models/default/', id=None):
        pretrain_path = os.path.join(path, 'pretrain')
        search_path = os.path.join(path, 'search')
        final_path = os.path.join(path, 'final', str(id))

        self._load_pretrain(pretrain_path)
        self._load_search(search_path)
        self._load_final(final_path)

        self._load_epoch('pretrain', pretrain_path)
        self._load_epoch('search', search_path)
        self._load_epoch('final', final_path)

        self._load_accuracy('pretrain', pretrain_path)
        self._load_accuracy('final', final_path)


    def _load_pretrain(self, path='saved_models/default/pretrain/'):
        try:
            filename = os.path.join(path, 'model')
            self.model.load_state_dict(torch.load(filename))

        except FileNotFoundError:
            pass


    def _load_search(self, path='saved_models/default/search/'):
        try:
            with open(os.path.join(path, 'queue.json')) as f:
                self.queue = json.load(f)
                self.queue = [(torch.tensor(masks[0], dtype=torch.uint8), [torch.tensor(m, dtype=torch.uint8) for m in masks[1]]) for masks in self.queue]
            with open(os.path.join(path, 'queue_obj.json')) as f:
                self.queue_obj = json.load(f)
                self.queue_obj = [tuple(obj) for obj in self.queue_obj]
            with open(os.path.join(path, 'pareto_front.json')) as f:
                self.pareto_front = json.load(f)
                self.pareto_front = [(torch.tensor(masks[0], dtype=torch.uint8), [torch.tensor(m, dtype=torch.uint8) for m in masks[1]]) for masks in self.pareto_front]
            with open(os.path.join(path, 'pareto_front_obj.json')) as f:
                self.pareto_front_obj = json.load(f)
                self.pareto_front_obj = [tuple(obj) for obj in self.pareto_front_obj]

        except FileNotFoundError:
            self.queue = []
            self.queue_acc = []
            self.pareto_front = []
            self.pareto_front_obj = []


    def _load_final(self, path='saved_models/default/final/'):
        try:
            with open(os.path.join(path, 'masks.json'), 'r') as f:
                self.finalmodel_mask = json.load(f)
            self.finalmodel_mask = [torch.tensor(self.finalmodel_mask[0], dtype=torch.uint8), [torch.tensor(m, dtype=torch.uint8) for m in self.finalmodel_mask[1]]]
            self.finalmodel = [self.submodel(self.finalmodel_mask[0], self.finalmodel_mask[1][task], task) for task in range(self.num_tasks)]
            self.finalmodel = [nn.DataParallel(m).to(self.device) for m in self.finalmodel]

            for t, model in enumerate(self.finalmodel):
                filename = os.path.join(path, 'model{}'.format(t))
                model.load_state_dict(torch.load(filename))

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
