import torch.nn as nn
import torch.optim as optim
import os
from .multi_task_single_obj import MultiTaskSingleObjectiveAgent
from .single_task_multi_obj import SingleTaskMultiObjectiveAgent


class MultiTaskMultiObjectiveAgent(MultiTaskSingleObjectiveAgent, SingleTaskMultiObjectiveAgent):
    def __init__(self, architecture, search_space, task_info):
        super(MultiTaskMultiObjectiveAgent, self).__init__(architecture, search_space, task_info)


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
            self.finalmodel = [self.submodel(self.finalmodel_mask, task) for task in range(self.num_tasks)]
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


""" Inherit from Multi Task Single Objective Agent """

MultiTaskMultiObjectiveAgent._pretrain = MultiTaskSingleObjectiveAgent._pretrain
MultiTaskMultiObjectiveAgent._eval_model = MultiTaskSingleObjectiveAgent._eval_model
MultiTaskMultiObjectiveAgent._eval_final = MultiTaskSingleObjectiveAgent._eval_final
MultiTaskMultiObjectiveAgent._eval = MultiTaskSingleObjectiveAgent._eval
MultiTaskMultiObjectiveAgent._save_final = MultiTaskSingleObjectiveAgent._save_final
MultiTaskMultiObjectiveAgent._load_final = MultiTaskSingleObjectiveAgent._load_final

""" Inherit from Single Task Multi Objective Agent """

MultiTaskMultiObjectiveAgent._init = SingleTaskMultiObjectiveAgent._init
MultiTaskMultiObjectiveAgent.train = SingleTaskMultiObjectiveAgent.train
MultiTaskMultiObjectiveAgent._search = SingleTaskMultiObjectiveAgent._search
MultiTaskMultiObjectiveAgent._save_search = SingleTaskMultiObjectiveAgent._save_search
MultiTaskMultiObjectiveAgent._load_search = SingleTaskMultiObjectiveAgent._load_search
MultiTaskMultiObjectiveAgent.load = SingleTaskMultiObjectiveAgent.load

"""
Inherit from both classes:
* eval
* _save_pretrain
* _save_epoch
* _save_accuracy
* _load_pretrain
* _load_epoch
* _load_accuracy
"""