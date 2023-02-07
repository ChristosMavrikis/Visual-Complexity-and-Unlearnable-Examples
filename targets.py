import time
import torch
import torchvision
import numpy as np
import skimage
from skimage.filters.rank import entropy
from torchvision import datasets, transforms
from AddGaussianNoise import AddGaussianNoise
import pyinform
from AddNoise import AddNoise
import imgaug.augmenters as iaa

train_transform= transforms.Compose ([
                            #np.array,
                            #iaa.JpegCompression(compression=30).augment_image,
                            #transforms.ToPILImage(),
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            np.array,
                            iaa.JpegCompression(compression=30).augment_image,
                            transforms.ToPILImage(),

                            #transforms.Grayscale(1),
                            transforms.ToTensor()])




data_train = torchvision.datasets.ImageNet('/scratch/data_share_ChDi/Imagenet-100', 'train',transform=train_transform)
print(data_train.targets)
