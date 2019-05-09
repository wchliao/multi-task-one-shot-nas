import numpy as np
import torch


class MaskSampler:
    def __init__(self, mask_size):
        self.mask_size = mask_size
        self.device_count = torch.cuda.device_count()


    def make_batch(self, masks):
        return torch.stack([masks for _ in range(self.device_count)])


    def rand(self, dropout=0.5, batch=False):
        probs = torch.rand(self.mask_size)
        masks = probs.gt(dropout)

        if batch:
            return self.make_batch(masks)
        else:
            return masks


    def ones(self, batch=False):
        masks = torch.ones(self.mask_size, dtype=torch.uint8)

        if batch:
            return self.make_batch(masks)
        else:
            return masks


    def zeros(self, batch=False):
        masks = torch.zeros(self.mask_size, dtype=torch.uint8)

        if batch:
            return self.make_batch(masks)
        else:
            return masks


    def mutate(self, masks, mutate_prob=0.05, batch=False):
        probs = np.random.rand(self.mask_size)
        mutates = probs < mutate_prob

        if not mutates.any():
            must_select_idx = np.random.choice(self.mask_size)
            probs[must_select_idx] = True

        new_masks = torch.tensor([mask if not mutate else not mask for mask, mutate in zip(masks, mutates)], dtype=torch.uint8)

        if batch:
            return self.make_batch(new_masks)
        else:
            return new_masks


def masks2str(masks):
    s = ''
    for m in masks:
        if m:
            s += '1'
        else:
            s += '0'

    return s


def _conv_size(in_channels, out_channels, kernel_size):
    return in_channels * out_channels * kernel_size * kernel_size


def _batchnorm_size(num_channels):
    return num_channels * 4


def _operation_size(in_channels, out_channels, operation, batchnorm=True):
    kernel_size = operation.kernel_size
    hidden_channels = operation.expansion * in_channels

    size = _conv_size(in_channels, hidden_channels, 1)
    size += _conv_size(hidden_channels, 1, kernel_size)
    size += _conv_size(hidden_channels, out_channels, 1)

    if batchnorm:
        size += _batchnorm_size(hidden_channels) * 2
        size += _batchnorm_size(out_channels)

    return size


def _fc_size(in_dims, out_dims):
    return in_dims * out_dims + out_dims


class ModelSize:
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
        masks = masks.clone()
        masks = masks.view(-1, len(self.ops))

        size = 0
        in_channels = self.in_channels

        for out_channels, layer_masks, must_select in zip(self.out_channels, masks, self.must_select):
            if must_select and not layer_masks.any():
                layer_masks[0] = 1
            for op, mask in zip(self.ops, layer_masks):
                if mask:
                    size += _operation_size(in_channels, out_channels, op)
            in_channels = out_channels

        size += _fc_size(in_channels, self.num_classes)

        return size


def _pareto_dominate(pa, pb):
    for a, b in zip(pa, pb):
        if a < b:
            return False
    return True


def _pareto_front(points):
    pareto_points = set()
    dominated_points = set()

    while len(points) > 0:
        point = points.pop()
        pareto = True
        remove_points = []

        for p in points:
            if _pareto_dominate(point, p):
                remove_points.append(p)
            elif pareto and _pareto_dominate(p, point):
                pareto = False

        if pareto:
            pareto_points.add(point)
        else:
            dominated_points.add(point)

        for p in remove_points:
            points.remove(p)
            dominated_points.add(p)

    pareto_points = [tuple(p) for p in pareto_points]
    dominated_points = [tuple(p) for p in dominated_points]

    return pareto_points, dominated_points


def pareto_front(points):
    tmp_points = points.copy()
    pareto_points, tmp_points = _pareto_front(tmp_points)
    idx = [points.index(p) for p in pareto_points]

    return pareto_points, idx


def pareto_front_full(points, num=None):
    if num is None:
        num = len(points)

    results = []
    tmp_points = points.copy()

    while len(results) < num:
        pareto_points, tmp_points = _pareto_front(tmp_points)
        results += pareto_points

    idx = [points.index(p) for p in results]

    return results[:num], idx[:num]