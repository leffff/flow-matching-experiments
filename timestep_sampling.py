import scipy.stats as stats
import numpy as np
import torch


class Uniform:
    @staticmethod
    def sample(num_samples):
        return torch.rand((num_samples,))


class LogitNormal:
    @staticmethod
    def sample(num_samples):
        return torch.randn((num_samples,)).sigmoid()


def exponential_pdf(x, a):
    C = a / (np.exp(a) - 1)
    return C * np.exp(a * x)


class ExponentialPDF(stats.rv_continuous):
    def _pdf(self, x, a):
        return exponential_pdf(x, a)


class UShaped:
    def __init__(self, a):
        self.a = a
        self.exponential_distribution = ExponentialPDF(a=0, b=1, name='ExponentialPDF')

    def sample(self, num_samples):
        t = self.exponential_distribution.rvs(size=num_samples, a=self.a)
        t = torch.from_numpy(t).float()
        t = torch.cat([t, 1 - t], dim=0)
        t = t[torch.randperm(t.shape[0])]
        t = t[:num_samples]

        t_min = 1e-5
        t_max = 1 - 1e-5

        t = t * (t_max - t_min) + t_min

        return t
