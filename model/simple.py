import torch.nn as nn
from .base import BaseModel, SubModel
from .core import Bottleneck


class SimpleModel(BaseModel):
    def __init__(self, architecture, search_space, in_channels, num_classes):
        super(SimpleModel, self).__init__(architecture, search_space, in_channels, num_classes)

        self.architecture = architecture
        self.search_size = len(search_space)
        self.num_layers = len(architecture)

        # Operations

        self.ops = []
        self.must_select = []

        for configs in architecture:
            if in_channels != configs.out_channels or configs.stride > 1:
                self.must_select.append(True)
            else:
                self.must_select.append(False)

            layer_ops = [Bottleneck(in_channels, configs.out_channels, op.kernel_size, op.expansion, configs.stride, bn_running_stats=True) for op in search_space]
            self.ops.append(nn.ModuleList(layer_ops))
            in_channels = configs.out_channels

        self.ops = nn.ModuleList(self.ops)

        # Decoder

        self.relu = nn.ReLU6(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.2, inplace=True)
        self.fc = nn.ModuleList([nn.Linear(architecture[-1].out_channels, c) for c in num_classes])


    def forward(self, inputs, masks, task=0):
        masks = masks.view(-1, self.search_size)
        x = inputs

        for layer_ops, layer_masks, must_select in zip(self.ops, masks, self.must_select):
            if not must_select and not layer_masks.any():
                continue
            if must_select and not layer_masks.any():
                layer_masks[0] = 1
            x = sum([op(x) for op, mask in zip(layer_ops, layer_masks) if mask])

        x = self.relu(x)
        x = self.avg_pool(x)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)
        x = self.fc[task](x)

        return x


    def submodel(self, masks, task=0):
        masks = masks.view(-1, self.search_size)

        ops = []

        for layer_ops, layer_masks, must_select in zip(self.ops, masks, self.must_select):
            if not must_select and not layer_masks.any():
                continue
            if must_select and not layer_masks.any():
                layer_masks[0] = 1

            ops.append(nn.ModuleList([op for op, mask in zip(layer_ops, layer_masks) if mask]))

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
