import os
import torch
import time
import collections
from PIL import Image, ImageFilter
import imgaug.augmenters as iaa

import random
import matplotlib.pyplot as plt
import matplotlib
from random import randint
import numpy as np
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

train_trans= transforms.Compose([              
                            #transforms.CenterCrop(256),
                            transforms.Resize((224,224)),
                            # transforms.Resize((56,56)),
                            #transforms.RandomResizedCrop((56,56)),
                            #transforms.RandomHorizontalFlip(),
                            #transforms.Grayscale(1),
                           transforms.ToTensor()])





train_jpeg= transforms.Compose([
                            #transforms.CenterCrop(256),
                              np.array,
                          iaa.JpegCompression(compression=60).augment_image,
                          transforms.ToPILImage(),
                            
                            transforms.Resize((224,224)),
                            # transforms.Resize((56,56)),
                            #transforms.RandomResizedCrop((56,56)),
                            #transforms.RandomHorizontalFlip(),
                            #transforms.Grayscale(1),
                           transforms.ToTensor()])


    # np.array,
                          #iaa.JpegCompression(compression=60).augment_image,
                          #transforms.ToPILImage(),


class ImageNetMini(datasets.ImageNet):
    def __init__(self, root, split='train', **kwargs):
        super(ImageNetMini, self).__init__(root, split=split, **kwargs)
        self.new_targets = []
        self.new_images = []
        #l = random.sample(range(1,1000),100)
        #l = list()
        #l= [0,510,134,283,346,153,30,555] #similar cifar classes
        for i, (file, cls_id) in enumerate(self.imgs):
            #if cls_id >99 and cls_id<=199:
            #if cls_id in l:
            #if cls_id <=99:
                self.new_targets.append(cls_id)
                self.new_images.append((file, cls_id))
        self.imgs = self.new_images
        self.targets = self.new_targets
        self.samples = self.imgs
        print(len(self.samples))
        print(len(self.targets))
        return





class PoisonImageNetMini(ImageNetMini):
    def __init__(self, root, split, poison_rate=1.0, seed=0,
                 perturb_tensor_filepath=None, **kwargs):
        super(PoisonImageNetMini, self).__init__(root=root, split=split, **kwargs)
        np.random.seed(seed)
        self.poison_rate = poison_rate
        self.perturb_tensor = torch.load(perturb_tensor_filepath)
        self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu').numpy()

        # Random Select Poison Targets
        targets = list(range(0, len(self)))
        self.poison_samples_idx = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())
        self.poison_samples = collections.defaultdict(lambda: False)
        self.poison_class = []
        for idx in self.poison_samples_idx:
            self.poison_samples[idx] = True

        print(self.perturb_tensor.shape)
        print('Poison samples: %d/%d' % (len(self.poison_samples), len(self)))

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        #sample = np.array(transforms.RandomResizedCrop(224)(sample)).astype(np.float32)
        sample = np.array(transforms.Resize((224,224))(sample)).astype(np.float32)
        #sample = np.array(transforms.RandomResizedCrop(56)(sample)).astype(np.float32)
        if self.poison_samples[index]:
           #noise = self.perturb_tensor[target]
            noise = self.perturb_tensor[index]
            sample = sample + noise
            sample = np.clip(sample, 0, 255)
        sample = sample.astype(np.uint8)
        sample = Image.fromarray(sample).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target




#poison = np.load('poison_targets.npy')
#poison = torch.load('/scratch/data_share_ChDi/Unlearnable-Examples-main/experiments/imagenet-mini/min-min_samplewise/ImageNetMini-eps=8-se=0.1-base_version=resnet18/perturbation.pt')


#poison_8= torch.load('/scratch/data_share_ChDi/Unlearnable-Examples-main/experiments/imagenet-mini/min-min_samplewise/ImageNetMini-eps=8-se=0.1-base_version=resnet18-56x56-rand/perturbation.pt')
#poison_16= torch.load('/scratch/data_share_ChDi/Unlearnable-Examples-main/experiments/imagenet-mini/min-min_samplewise/ImageNetMini-eps=16-se=0.1-base_version=resnet18-56x56-rand/perturbation.pt')
#poison_32= torch.load('/scratch/data_share_ChDi/Unlearnable-Examples-main/experiments/imagenet-mini/min-min_samplewise/ImageNetMini-eps=32-se=0.1-base_version=resnet18-56x56-rand/perturbation.pt')



#poison = torch.load('/scratch/data_share_ChDi/Unlearnable-Examples-main/experiments/imagenet-mini/min-min_samplewise/ImageNetMini-eps=8-se=0.1-base_version=resnet18/perturbation.pt')
#poison = torch.load('/ceph/csedu-scratch/project/mavrikis/data_share_ChDi/Unlearnable-Examples-main/experiments/coco/min-min_samplewise/MCOCO-eps=8-se=0.1-base_version=resnet18$/perturbation.pt')
#data_train = torchvision.datasets.ImageNet('/scratch/data_share_ChDi/Imagenet-100', 'train')
#poison = torch.load('/scratch/data_share_ChDi/Unlearnable-Examples-main/experiments/imagenet-mini/min-min_samplewise/ImageNetMini-eps=16-se=0.1-base_version=resnet18/perturbation.pt')
#print(data_train[0][0])
#print(data_train.imgs[0])

#print(poison)


test_dataset_original = PoisonImageNetMini(root='/ceph/csedu-scratch/project/mavrikis/data_share_ChDi/Random-ImageNet', split='train', seed=0,transform=train_trans, poison_rate=1.0,perturb_tensor_filepath='/ceph/csedu-scratch/project/mavrikis/data_share_ChDi/Unlearnable-Examples-main/experiments/imagenet-mini/min-min_samplewise/ImageNetMini-eps=8-se=0.1-base_version=resnet18-rand-jpeg-30/perturbation.pt')

test_dataset_original_1 = PoisonImageNetMini(root='/ceph/csedu-scratch/project/mavrikis/data_share_ChDi/Random-ImageNet', split='train', seed=0,transform=train_trans, poison_rate=1.0,perturb_tensor_filepath='/ceph/csedu-scratch/project/mavrikis/data_share_ChDi/Unlearnable-Examples-main/experiments/imagenet-mini/min-min_samplewise/ImageNetMini-eps=16-se=0.1-base_version=resnet18-rand-jpeg-30/perturbation.pt')


test_dataset_original_2 = PoisonImageNetMini(root='/ceph/csedu-scratch/project/mavrikis/data_share_ChDi/Random-ImageNet', split='train', seed=0,transform=train_trans, poison_rate=1.0,perturb_tensor_filepath='/ceph/csedu-scratch/project/mavrikis/data_share_ChDi/Unlearnable-Examples-main/experiments/imagenet-mini/min-min_samplewise/ImageNetMini-eps=32-se=0.1-base_version=resnet18-rand-jpeg-30/perturbation.pt')

test_dataset_jpeg = PoisonImageNetMini(root='/ceph/csedu-scratch/project/mavrikis/data_share_ChDi/Random-ImageNet', split='train', seed=0,transform=train_trans, poison_rate=1.0,perturb_tensor_filepath='/ceph/csedu-scratch/project/mavrikis/data_share_ChDi/Unlearnable-Examples-main/experiments/imagenet-mini/min-min_samplewise/ImageNetMini-eps=8-se=0.1-base_version=resnet18-rand-jpeg-60/perturbation.pt')


test_dataset_jpeg_1 = PoisonImageNetMini(root='/ceph/csedu-scratch/project/mavrikis/data_share_ChDi/Random-ImageNet', split='train', seed=0,transform=train_trans, poison_rate=1.0,perturb_tensor_filepath='/ceph/csedu-scratch/project/mavrikis/data_share_ChDi/Unlearnable-Examples-main/experiments/imagenet-mini/min-min_samplewise/ImageNetMini-eps=16-se=0.1-base_version=resnet18-rand-jpeg-60/perturbation.pt')


test_dataset_jpeg_2 = PoisonImageNetMini(root='/ceph/csedu-scratch/project/mavrikis/data_share_ChDi/Random-ImageNet', split='train', seed=0,transform=train_trans, poison_rate=1.0,perturb_tensor_filepath='/ceph/csedu-scratch/project/mavrikis/data_share_ChDi/Unlearnable-Examples-main/experiments/imagenet-mini/min-min_samplewise/ImageNetMini-eps=32-se=0.1-base_version=resnet18-rand-jpeg-60/perturbation.pt')





test_dataset_gray = PoisonImageNetMini(root='/ceph/csedu-scratch/project/mavrikis/data_share_ChDi/Random-ImageNet', split='train', seed=0,transform=train_trans, poison_rate=1.0,perturb_tensor_filepath='/ceph/csedu-scratch/project/mavrikis/data_share_ChDi/Unlearnable-Examples-main/experiments/imagenet-mini/min-min_samplewise/ImageNetMini-eps=8-se=0.1-base_version=resnet18-rand-jpeg-90/perturbation.pt')

test_dataset_gray_1 = PoisonImageNetMini(root='/ceph/csedu-scratch/project/mavrikis/data_share_ChDi/Random-ImageNet', split='train', seed=0,transform=train_trans, poison_rate=1.0,perturb_tensor_filepath='/ceph/csedu-scratch/project/mavrikis/data_share_ChDi/Unlearnable-Examples-main/experiments/imagenet-mini/min-min_samplewise/ImageNetMini-eps=16-se=0.1-base_version=resnet18-rand-jpeg-90/perturbation.pt')


test_dataset_gray_2 = PoisonImageNetMini(root='/ceph/csedu-scratch/project/mavrikis/data_share_ChDi/Random-ImageNet', split='train', seed=0,transform=train_trans, poison_rate=1.0,perturb_tensor_filepath='/ceph/csedu-scratch/project/mavrikis/data_share_ChDi/Unlearnable-Examples-main/experiments/imagenet-mini/min-min_samplewise/ImageNetMini-eps=32-se=0.1-base_version=resnet18-rand-jpeg-90/perturbation.pt')





#test_dataset_1 = PoisonImageNetMini(root='/ceph/csedu-scratch/project/mavrikis/data_share_ChDi/Random-ImageNet', split='train', seed=0,transform=train_trans, poison_rate=1.0,perturb_tensor_filepath='/scratch/data_share_ChDi/Unlearnable-Examples-main/experiments/imagenet-mini/min-min_samplewise/ImageNetMini-eps=16-se=0.1-base_version=resnet18-112x112-rand/perturbation.pt')
#test_dataset_2 = PoisonImageNetMini(root='/ceph/csedu-scratch/project/mavrikis/data_share_ChDi/Random-ImageNet', split='train', seed=0,transform=train_trans, poison_rate=1.0,perturb_tensor_filepath='/scratch/data_share_ChDi/Unlearnable-Examples-main/experiments/imagenet-mini/min-min_samplewise/ImageNetMini-eps=32-se=0.1-base_version=resnet18-112x112-rand/perturbation.pt')





train_dataset = datasets.ImageNet(root='/ceph/csedu-scratch/project/mavrikis/data_share_ChDi/Random-ImageNet', split='train',transform=train_trans)
train_dataset_jpeg = datasets.ImageNet(root='/ceph/csedu-scratch/project/mavrikis/data_share_ChDi/Random-ImageNet', split='train',transform=train_jpeg)


i=0
while ( i < 6):
    j = randint(0,25000)
    tmp_original, l_original = train_dataset[j]
    tmp_jpeg, l_jpeg = train_dataset_jpeg[j]

    tmp,l = test_dataset_original[j]

    fig = plt.figure(figsize=(8, 8), dpi=80,facecolor='w', edgecolor='k')
    npimg =  tmp_original
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    plt.savefig('rand-original-224x224-'+str(i)+'.png')
    
    fig = plt.figure(figsize=(8, 8), dpi=80,facecolor='w', edgecolor='k')
    npimg =  tmp_jpeg
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    plt.savefig('rand-jpeg60-224x224-'+str(i)+'.png')

    #original-e8
    fig = plt.figure(figsize=(8, 8), dpi=80,facecolor='w', edgecolor='k')
    npimg =  tmp
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    plt.savefig('rand-jpeg-30-224x224-e8-'+str(i)+'.png')
    
    npimg =  test_dataset_original.perturb_tensor[j]
    plt.imshow(np.transpose(npimg, (1, 0, 2)))
    plt.show()
    plt.savefig('noise-jpeg-30-224x224-e8-'+str(i)+'.png')
    
    #original-e16
    tmp,l = test_dataset_original_1[j]
    fig = plt.figure(figsize=(8, 8), dpi=80,facecolor='w', edgecolor='k')
    npimg =  tmp
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    plt.savefig('rand-jpeg-30-224x224-e16-'+str(i)+'.png')

    npimg =  test_dataset_original_1.perturb_tensor[j] #change
    plt.imshow(np.transpose(npimg, (1, 0, 2)))
    plt.show()
    plt.savefig('noise-jpeg-30-224x224-e16-'+str(i)+'.png')
    
    #original-e32
    tmp,l = test_dataset_original_2[j]
    fig = plt.figure(figsize=(8, 8), dpi=80,facecolor='w', edgecolor='k')
    npimg =  tmp
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    plt.savefig('rand-jpeg-30-224x224-e32-'+str(i)+'.png')

    npimg =  test_dataset_original_2.perturb_tensor[j] #change
    plt.imshow(np.transpose(npimg, (1, 0, 2)))
    plt.show()
    plt.savefig('noise-jpeg-30-224x224-e32-'+str(i)+'.png')

    
    #gray-e8
    tmp,l = test_dataset_gray[j]
    fig = plt.figure(figsize=(8, 8), dpi=80,facecolor='w', edgecolor='k')
    npimg =  tmp
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    plt.savefig('rand-jpeg-90-224x224-e8-'+str(i)+'.png')

    npimg =  test_dataset_gray.perturb_tensor[j]
    plt.imshow(np.transpose(npimg, (1, 0, 2)))
    plt.show()
    plt.savefig('noise-jpeg-90-224x224-e8-'+str(i)+'.png')

    #original-e16
    tmp,l = test_dataset_gray_1[j]
    fig = plt.figure(figsize=(8, 8), dpi=80,facecolor='w', edgecolor='k')
    npimg =  tmp
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    plt.savefig('rand-jpeg-90-224x224-e16-'+str(i)+'.png')

    npimg =  test_dataset_gray_1.perturb_tensor[j] #change
    plt.imshow(np.transpose(npimg, (1, 0, 2)))
    plt.show()
    plt.savefig('noise-jpeg-90-224x224-e16-'+str(i)+'.png')

    #original-e32
    tmp,l = test_dataset_gray_2[j]
    fig = plt.figure(figsize=(8, 8), dpi=80,facecolor='w', edgecolor='k')
    npimg =  tmp
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    plt.savefig('rand-jpeg-90-224x224-e32-'+str(i)+'.png')

    npimg =  test_dataset_gray_2.perturb_tensor[j] #change
    plt.imshow(np.transpose(npimg, (1, 0, 2)))
    plt.show()
    plt.savefig('noise-jpeg-90-224x224-e32-'+str(i)+'.png')


    #gray-e8
    tmp,l = test_dataset_jpeg[j]
    fig = plt.figure(figsize=(8, 8), dpi=80,facecolor='w', edgecolor='k')
    npimg =  tmp
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    plt.savefig('rand-jpeg-60-224x224-e8-'+str(i)+'.png')

    npimg =  test_dataset_jpeg.perturb_tensor[j]
    plt.imshow(np.transpose(npimg, (1, 0, 2)))
    plt.show()
    plt.savefig('noise-jpeg-60-224x224-e8-'+str(i)+'.png')

    #original-e16
    tmp,l = test_dataset_jpeg_1[j]
    fig = plt.figure(figsize=(8, 8), dpi=80,facecolor='w', edgecolor='k')
    npimg =  tmp
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    plt.savefig('rand-jpeg-60-224x224-e16-'+str(i)+'.png')

    npimg =  test_dataset_jpeg_1.perturb_tensor[j] #change
    plt.imshow(np.transpose(npimg, (1, 0, 2)))
    plt.show()
    plt.savefig('noise-jpeg-60-224x224-e16-'+str(i)+'.png')

    #original-e32
    tmp,l = test_dataset_jpeg_2[j]
    fig = plt.figure(figsize=(8, 8), dpi=80,facecolor='w', edgecolor='k')
    npimg =  tmp
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    plt.savefig('rand-jpeg-60-224x224-e32-'+str(i)+'.png')

    npimg =  test_dataset_jpeg_2.perturb_tensor[j] #change
    plt.imshow(np.transpose(npimg, (1, 0, 2)))
    plt.show()
    plt.savefig('noise-jpeg-60-224x224-e32-'+str(i)+'.png')

    i=i+1

