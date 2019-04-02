import torch


class MaskSampler:
    def __init__(self, mask_size, controller=None):
        self.mask_size = mask_size
        self.controller = controller
        self.device_count = torch.cuda.device_count()


    def rand(self, batch=True):
        masks = torch.rand(self.mask_size)

        if batch:
            return torch.stack([masks for _ in range(self.device_count)])
        else:
            return masks


    def ones(self, batch=True):
        masks = torch.ones(self.mask_size)

        if batch:
            return torch.stack([masks for _ in range(self.device_count)])
        else:
            return masks


    def sample(self, task=None, sample_best=False, grad=True, batch=True):
        if grad:
            masks, log_probs = self.controller.sample(task=task, sample_best=sample_best)
        else:
            with torch.no_grad():
                masks, log_probs = self.controller.sample(task=task, sample_best=sample_best)

        if batch:
            masks = torch.stack([masks for _ in range(self.device_count)])

        return masks, log_probs


class BaseModelSize:
    def __init__(self, architecture, search_space, in_channels, num_classes, batchnorm=True):
        self.in_channels = in_channels
        self.out_channels = [layer.out_channels for layer in architecture]
        self.ops = search_space
        self.num_classes = num_classes
        self.batchnorm = batchnorm

        self.must_select = []
        for layer in architecture:
            if in_channels != layer.out_channels or layer.stride > 1:
                self.must_select.append(True)
            else:
                self.must_select.append(False)

            in_channels = layer.out_channels


    def compute(self, masks):
        raise NotImplementedError


    def _conv_size(self, in_channels, out_channels, kernel_size):
        return in_channels * out_channels * kernel_size * kernel_size


    def _batchnorm_size(self, num_channels):
        return num_channels * 4


    def _operation_size(self, in_channels, out_channels, operation):
        kernel_size = operation.kernel_size
        hidden_channels = operation.expansion * in_channels

        size = self._conv_size(in_channels, hidden_channels, 1)
        size += self._conv_size(hidden_channels, 1, kernel_size)
        size += self._conv_size(hidden_channels, out_channels, 1)

        if self.batchnorm:
            size += self._batchnorm_size(hidden_channels) * 2
            size += self._batchnorm_size(out_channels)

        return size


    def _fc_size(self, in_dims, out_dims):
        return in_dims * out_dims + out_dims


class SingleModelSize(BaseModelSize):
    def __init__(self, architecture, search_space, in_channels, num_classes, batchnorm=True):
        super(SingleModelSize, self).__init__(architecture, search_space, in_channels, num_classes, batchnorm)


    def compute(self, masks):
        masks = masks.view(-1, len(self.ops))

        size = 0
        in_channels = self.in_channels

        for out_channels, layer_masks, must_select in zip(self.out_channels, masks, self.must_select):
            if must_select and layer_masks.sum() == 0:
                layer_masks[0] = 1
            for op, mask in zip(self.ops, layer_masks):
                if mask:
                    size += self._operation_size(in_channels, out_channels, op)
            in_channels = out_channels

        size += self._fc_size(in_channels, self.num_classes)

        return size


class MultiModelSize(BaseModelSize):
    def __init__(self, architecture, search_space, in_channels, num_classes, batchnorm=False):
        super(MultiModelSize, self).__init__(architecture, search_space, in_channels, num_classes, batchnorm)
        self.num_classes = sum(num_classes)


    def compute(self, masks):
        masks = [layer_masks.view(-1, len(self.ops)) for layer_masks in masks]
        for masks_t in masks:
            for layer_masks, must_select in zip(masks_t, self.must_select):
                if must_select and layer_masks.sum() == 0:
                    layer_masks[0] = 1
        masks = sum(masks)

        size = 0
        in_channels = self.in_channels

        for out_channels, layer_masks in zip(self.out_channels, masks):
            for op, mask in zip(self.ops, layer_masks):
                if mask:
                    size += self._operation_size(in_channels, out_channels, op)
            in_channels = out_channels

        size += self._fc_size(in_channels, self.num_classes)

        return size
