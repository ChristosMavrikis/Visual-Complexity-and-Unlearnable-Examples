import torch
import numpy as np

class AddNoise(object):
    def __init__(self, minimum=0., maximum=1.):
        self.maximum = maximum
        self.minimum = minimum

    def __call__(self, tensor):
        noise = np.random.uniform(self.minimum, self.maximum, tensor.size())
        noise = torch.tensor(noise)
        return tensor + noise


