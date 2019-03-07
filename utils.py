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
