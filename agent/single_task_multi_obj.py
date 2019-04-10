import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import numpy as np
from utils import MaskSampler, masks2str, pareto_front, pareto_front_full
from .base import BaseMultiObjectiveAgent
from .single_task_single_obj import SingleTaskSingleObjectiveAgent


class SingleTaskMultiObjectiveAgent(SingleTaskSingleObjectiveAgent, BaseMultiObjectiveAgent):
    def __init__(self, architecture, search_space, task_info):
        super(SingleTaskMultiObjectiveAgent, self).__init__(architecture, search_space, task_info)


    def _init(self):
        self.submodel = self.model.submodel
        self.mask_sampler = MaskSampler(mask_size=self.model.mask_size)
        self.model = nn.DataParallel(self.model).to(self.device)

        # Record

        self.epoch = {'pretrain': 0, 'search': 0, 'final': 0}
        self.accuracy = {'pretrain': [], 'final': []}

        # Search

        self.obj_dict_valid = {}
        self.obj_dict_test = {}
        self.queue = []
        self.queue_obj = []
        self.pareto_front = []
        self.pareto_front_obj = []

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


    def _search(self,
                valid_data,
                configs,
                save_model=False,
                path='saved_models/default/search/',
                verbose=False
                ):

        # Initalization

        if self.epoch['search'] == 0:
            self.queue = [self.mask_sampler.rand(dropout=i/(configs.num_samples-1)) for i in range(configs.num_samples)]
            self.queue_obj = []

            for masks in self.queue:
                masks_str = masks2str(masks)
                accuracy = self._eval_model(valid_data, masks)
                model_size = self.compute_model_size.compute(masks)
                self.obj_dict_valid[masks_str] = (accuracy, -model_size)
                self.queue_obj.append((accuracy, -model_size))

        # Search

        for epoch in range(self.epoch['search'], configs.num_epochs):
            generated = []
            generated_obj = []

            for old_masks in self.queue:
                new_masks = self.mask_sampler.mutate(old_masks, configs.mutate_prob)
                new_masks_str = masks2str(new_masks)

                if new_masks_str not in self.obj_dict_valid:
                    accuracy = self._eval_model(valid_data, new_masks)
                    model_size = self.compute_model_size.compute(new_masks)
                    self.obj_dict_valid[new_masks_str] = (accuracy, -model_size)

                generated.append(new_masks)
                generated_obj.append(self.obj_dict_valid[new_masks_str])

            candidates = self.queue + generated
            candidates_obj = self.queue_obj + generated_obj
            self.queue_obj, order = pareto_front_full(candidates_obj, num=configs.num_samples)
            self.queue = [candidates[i] for i in order]

            if verbose:
                print('[Search][Epoch {}]'.format(epoch + 1))

            if epoch % configs.save_epoch == 0:
                if save_model:
                    self._save_search(path)
                    self.epoch['search'] = epoch + 1
                    self._save_epoch('search', path)

        self.pareto_front_obj, order = pareto_front(self.queue_obj)
        self.pareto_front = [self.queue[i] for i in order]
        order = np.argsort([obj[0] for obj in self.pareto_front_obj])[::-1]
        self.pareto_front = [self.pareto_front[i] for i in order]
        self.pareto_front_obj = [self.pareto_front_obj[i] for i in order]

        if save_model:
            self._save_search(path)
            self.epoch['search'] = configs.num_epochs
            self._save_epoch('search', path)


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
            self.finalmodel = self.submodel(self.finalmodel_mask)
            self.finalmodel = nn.DataParallel(self.finalmodel).to(self.device)

        if save_model:
            self._save_model_size(path)

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


    def _save_model_size(self, path='saved_models/default/'):
        if not os.path.isdir(path):
            os.makedirs(path)
        filename = os.path.join(path, 'model_size.json')

        with open(filename, 'w') as f:
            json.dump(self.compute_model_size.compute(self.finalmodel_mask), f)


    def _save_search(self, path='saved_models/default/search/'):
        if not os.path.isdir(path):
            os.makedirs(path)

        with open(os.path.join(path, 'obj_dict_valid.json'), 'w') as f:
            json.dump(self.obj_dict_valid, f)
        with open(os.path.join(path, 'obj_dict_test.json'), 'w') as f:
            json.dump(self.obj_dict_test, f)
        with open(os.path.join(path, 'queue.json'), 'w') as f:
            json.dump([masks.tolist() for masks in self.queue], f)
        with open(os.path.join(path, 'queue_obj.json'), 'w') as f:
            json.dump(self.queue_obj, f)
        with open(os.path.join(path, 'pareto_front.json'), 'w') as f:
            json.dump([masks.tolist() for masks in self.pareto_front], f)
        with open(os.path.join(path, 'pareto_front_obj.json'), 'w') as f:
            json.dump(self.pareto_front_obj, f)


    def _load_search(self, path='saved_models/default/search/'):
        try:
            with open(os.path.join(path, 'obj_dict_valid.json')) as f:
                self.obj_dict_valid = json.load(f)
            with open(os.path.join(path, 'obj_dict_test.json')) as f:
                self.obj_dict_test = json.load(f)
            with open(os.path.join(path, 'queue.json')) as f:
                self.queue = json.load(f)
                self.queue = [torch.tensor(masks, dtype=torch.uint8) for masks in self.queue]
            with open(os.path.join(path, 'queue_obj.json')) as f:
                self.queue_obj = json.load(f)
                self.queue_obj = [tuple(obj) for obj in self.queue_obj]
            with open(os.path.join(path, 'pareto_front.json')) as f:
                self.pareto_front = json.load(f)
                self.pareto_front = [torch.tensor(masks, dtype=torch.uint8) for masks in self.pareto_front]
            with open(os.path.join(path, 'pareto_front_obj.json')) as f:
                self.pareto_front_obj = json.load(f)
                self.pareto_front_obj = [tuple(obj) for obj in self.pareto_front_obj]

        except FileNotFoundError:
            self.queue = []
            self.queue_acc = []
            self.pareto_front = []
            self.pareto_front_obj = []
