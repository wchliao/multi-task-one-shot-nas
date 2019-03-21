import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, architecture, search_space, in_channels, num_classes, bn_running_stats=True):
        super(BaseModel, self).__init__()

    def forward(self, inputs, masks, task):
        raise NotImplementedError

    def submodel(self, masks):
        raise NotImplementedError

    @property
    def mask_size(self):
        raise NotImplementedError


class SubModel(nn.Module):
    def __init__(self, ops, activation, decoder):
        super(SubModel, self).__init__()
        self.ops = ops
        self.activation = activation
        self.decoder = decoder


    def forward(self, inputs):
        x = inputs
        for layer_ops in self.ops:
            x = sum([op(x) for op in layer_ops])

        x = self.activation(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)

        return x
