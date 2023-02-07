import time
import torch
import torchvision
import numpy as np
import skimage
from skimage.filters.rank import entropy
from torchvision import datasets, transforms
from AddGaussianNoise import AddGaussianNoise
import pyinform

def main():
    data_train = torchvision.datasets.ImageNet('/scratch/data_share_ChDi/Imagenet-100', 'train')
    x=0
    entropy = []
    cls = []
    y=False
    w=False
    i=0;
    while(y==False):
        t = []
        while(data_train.targets[i] == x and w==False):
        #print(i)
            if(w==False):
                tmp,l = data_train[i]
                convert_tensor =transforms.Resize((224,244))
                tmp = convert_tensor(tmp)
                convert_tensor =transforms.ToTensor()
                tmp = convert_tensor(tmp)
                convert_tensor = AddGaussianNoise(0.0,0.25)
                tmp = convert_tensor(tmp)
                #convert_tensor = transforms.ToTensor()
                #tmp = convert_tensor(tmp)
                tmp = tmp.mul(255).clamp_(0, 255).permute(1, 2, 0).to('cpu').numpy()
                tmp = tmp.astype(np.uint8)
                t.append(skimage.measure.shannon_entropy(tmp))
                i = i + 1
                if(i==len(data_train)):
                    w=True
                    i=i-1
        entropy.append(sum(t)/len(t))
        cls.append(x)
        if(x == 99): break
        x = x + 1

    arr = np.array(entropy)

