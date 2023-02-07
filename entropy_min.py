import time
import torch
import torchvision
import numpy as np
import skimage
from skimage.filters.rank import entropy
from torchvision import datasets, transforms
from AddGaussianNoise import AddGaussianNoise
from pyinform import Dist
from pyinform import entropy_rate
import imgaug.augmenters as iaa
from torchvision import datasets, transforms


train_transform =  transforms.Compose([  ##transforms.ToTensor(),
                            #AddGaussianNoise(0,0.2),
                            #np.array,
                            #iaa.JpegCompression(compression=30).augment_image,
                            #transforms.ToPILImage(),
                            #cluster for grayscale
                            transforms.Grayscale(3),
                            #transforms.RandomResizedCrop(224),
                            transforms.Resize(224),
                            #transforms.RandomEqualize(p=1),
                            #transforms.RandomAdjustSharpness(2, p=1),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor()])


def main():
    data_train = torchvision.datasets.ImageNet('/ceph/csedu-scratch/project/mavrikis/data_share_ChDi/Imagenet-100/', 'train',transform=train_transform)
    x=0
    entropy = []
    ent_rate = []
    excess_ent = []
    cls = []
    y=False
    w=False
    i=0
    print("here")
    while(y==False):
        t = []
        er = []
        ec = []
        w=False
        obs = 0
        while(data_train.targets[i]==x and w==False):
            print(i)
            if(w==False):
                tmp,l = data_train[i]
                #convert_tensor =transforms.Resize((224,224))
                ##tmp = convert_tensor(tmp)
                #convert_tensor =transforms.ToTensor()
                #tmp = convert_tensor(tmp)
                #histogram, bin_edges = np.histogram(tmp, bins=256, range=(0, 1))
                #D = Dist(histogram)
                #t.append(entropy(D))
                #er.append(entropy_rate(histogram,k=1))
                #excess_1 = (entropy_rate(histogram,k=2))-(entropy_rate(histogram,k=1))
                #excess_2 = (entropy_rate(histogram,k=4))-(entropy_rate(histogram,k=2))
                #excess_3 = (entropy_rate(histogram,k=5))-(entropy_rate(histogram,k=2))
                #excess_entropy = entropy_rate(histogram,k=2) - er[obs]
                #ec.append(excess_entropy)
                #convert_tensor = AddGaussianNoise(0.0,0.2)
                #tmp = convert_tensor(tmp)
                #convert_tensor = transforms.ToTensor()
                #tmp = convert_tensor(tmp)
                #noise=((torch.randn(3,224,224)* 0.2).numpy()*255).astype(np.int8)
                #noise=((torch.randn(224,224,3)* 0.2).numpy()*255).astype(np.int8)

                #tmp=tmp+noise
                #tmp = convert_tensor(tmp)
                tmp = tmp.mul(255).clamp_(0, 255).permute(1, 2, 0).to('cpu').numpy()
                tmp = tmp.astype(np.uint8)
                #tmp = tmp+noise
                t.append(skimage.measure.shannon_entropy(tmp))
                i = i + 1
                if(i==len(data_train)):
                    w=True
                    i=i-1
                #obs = obs + 1
        entropy.append(sum(t)/len(t))
        #ent_rate.append(sum(er)/len(er))
        #excess_ent.append(sum(ec)/len(ec))
        cls.append(x)
        #cls.append(x)
        if(x == 99): break
        x = x + 1
    print("Average entropy", entropy)


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    cost = (end - start) / 86400
    payload = "Running Cost %.2f Days \n" % cost
    #logger.info(payload)

