import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.color import rgb2gray
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

from skimage import data
from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.morphology import disk
from AddGaussianNoise import AddGaussianNoise

train_trans= [transforms.ToTensor(),
                            #AddGaussianNoise(0,0.0175),
                            transforms.ToPILImage(),
                            #transforms.Grayscale(3),
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor()],





data_train = torchvision.datasets.ImageNet('/scratch/data_share_ChDi/Imagenet-100', 'train',transform = train_trans)

print(data_train[0])
#print(skimage.measure.shannon_entropy(tmp))
