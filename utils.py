import torch


class MaskSampler:
    def __init__(self, mask_size, controller=None):
        self.mask_size = mask_size
        self.controller = controller
        self.device_count = torch.cuda.device_count()


    def rand(self):
        masks = torch.rand(self.mask_size)
        return torch.stack([masks for _ in range(self.device_count)])


    def ones(self):
        masks = torch.ones(self.mask_size)
        return torch.stack([masks for _ in range(self.device_count)])


    def sample(self, task=None, sample_best=False, grad=True):
        if grad:
            masks, log_probs = self.controller.sample(task=task, sample_best=sample_best)
        else:
            with torch.no_grad():
                masks, log_probs = self.controller.sample(task=task, sample_best=sample_best)

        return torch.stack([masks for _ in range(self.device_count)]), log_probs