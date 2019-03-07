import torch
import torch.nn as nn


class Controller(nn.Module):
    def __init__(self, num_outputs, num_tasks=None, hidden_size=128):
        super(Controller, self).__init__()

        if num_tasks is None:
            self.encoder = torch.ones(1, hidden_size)
        else:
            self.encoder = nn.Embedding(num_tasks, hidden_size)

        self.hidden = nn.Linear(hidden_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, num_outputs)


    def forward(self, task=None):
        if task is None:
            embed = self.encoder
        else:
            embed = self.encoder(task)

        x = self.hidden(embed)
        x = self.decoder(x)

        return x


    def sample(self, task=None, sample_best=False):
        if task is not None:
            task = torch.tensor(task)

        outputs = self.forward(task)
        probs = torch.sigmoid(outputs).view(-1)
        dist = torch.distributions.bernoulli.Bernoulli(probs)

        if sample_best:
            masks = probs.gt(0.5)
        else:
            masks = dist.sample()

        masks_probs = torch.tensor([p if mask else 1 - p for p, mask in zip(probs, masks)])
        log_probs = dist.log_prob(masks_probs)

        return masks, log_probs
