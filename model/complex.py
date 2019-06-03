import torch.nn as nn
from .base import SubModel
from .core import Bottleneck


class ComplexModel(nn.Module):
    def __init__(self, architecture, search_space, in_channels, num_classes):
        super(ComplexModel, self).__init__()

        self.architecture = architecture
        self.search_size = len(search_space)
        self.num_layers = len(architecture)
        num_tasks = len(num_classes)

        # Operations

        self.must_select = []

        for configs in architecture:
            if in_channels != configs.out_channels or configs.stride > 1:
                self.must_select.append(True)
            else:
                self.must_select.append(False)

        self.ops = []

        for _ in range(num_tasks + 1):
            task_ops = []
            in_channels_ = in_channels
            for configs in architecture:
                layer_ops = [Bottleneck(in_channels_, configs.out_channels, op.kernel_size, op.expansion, configs.stride, bn_running_stats=False) for op in search_space]
                task_ops.append(nn.ModuleList(layer_ops))
                in_channels_ = configs.out_channels

            task_ops = nn.ModuleList(task_ops)
            self.ops.append(task_ops)

        self.ops = nn.ModuleList(self.ops)

        # Decoder

        self.relu = nn.ReLU6(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.2, inplace=True)
        self.fc = nn.ModuleList([nn.Linear(architecture[-1].out_channels, c) for c in num_classes])


    def forward(self, inputs, shared_masks, task_masks, task=0):
        shared_masks = shared_masks.clone()
        shared_masks = shared_masks.view(-1, self.search_size)
        task_masks = task_masks.view(-1, self.search_size)
        x = inputs

        for layer_shared_ops, layer_task_ops, layer_shared_masks, layer_task_masks, must_select in zip(self.ops[-1], self.ops[task], shared_masks, task_masks, self.must_select):
            if not must_select and not layer_shared_masks.any() and not layer_task_masks.any():
                continue
            if must_select and not layer_shared_masks.any() and not layer_task_masks.any():
                layer_shared_masks[0] = 1
            x = sum([op(x) for op, mask in zip(layer_shared_ops, layer_shared_masks) if mask] + [op(x) for op, mask in zip(layer_task_ops, layer_task_masks) if mask])

        x = self.relu(x)
        x = self.avg_pool(x)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)
        x = self.fc[task](x)

        return x


    def submodel(self, shared_masks, task_masks, task=0):
        shared_masks = shared_masks.clone()
        shared_masks = shared_masks.view(-1, self.search_size)
        task_masks = task_masks.view(-1, self.search_size)

        ops = []

        for layer_shared_ops, layer_task_ops, layer_shared_masks, layer_task_masks, must_select in zip(self.ops[-1], self.ops[task], shared_masks, task_masks, self.must_select):
            if not must_select and not layer_shared_masks.any() and not layer_task_masks.any():
                continue
            if must_select and not layer_shared_masks.any() and not layer_task_masks.any():
                layer_shared_masks[0] = 1

            ops.append(nn.ModuleList([op for op, mask in zip(layer_shared_ops, layer_shared_masks) if mask] + [op for op, mask in zip(layer_task_ops, layer_task_masks) if mask]))

        ops = nn.ModuleList(ops)

        activation = nn.Sequential(
            self.relu,
            self.avg_pool,
            self.dropout
        )

        return SubModel(ops, activation, self.fc[task])


    @property
    def mask_size(self):
        return self.num_layers * self.search_size
