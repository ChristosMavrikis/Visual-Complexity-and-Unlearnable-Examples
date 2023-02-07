import time
import torch
import torchvision
import numpy as np
import skimage
from skimage.filters.rank import entropy
from torchvision import datasets, transforms
from AddGaussianNoise import AddGaussianNoise
import matplotlib.pyplot as plt

nb_bins = 256
count_r = np.zeros(nb_bins)
count_g = np.zeros(nb_bins)
count_b = np.zeros(nb_bins)


def main():
    data_train = torchvision.datasets.ImageNet('/scratch/data_share_ChDi/Imagenet-100', 'train')
    x=0
    entropy = []
    cls = []
    y=False
    w=False
    i=0
    nb_bins = 256
    count_r = np.zeros(nb_bins)
    count_g = np.zeros(nb_bins)
    count_b = np.zeros(nb_bins)

    while(y==False):
        t = []
        while(data_train.targets[i] == x and w==False):
        #print()
            w=False
            if(w==False):
                tmp,l = data_train[i]
                convert_tensor =transforms.Resize((224,244))
                tmp = convert_tensor(tmp)
                convert_tensor =transforms.ToTensor()
                tmp = convert_tensor(tmp)
                #convert_tensor = AddGaussianNoise(0.0,0.25)
                #tmp = convert_tensor(tmp)
                #tmp=tmp.numpy()
                
                hist_r = np.histogram(tmp[0], bins=nb_bins, range=[0, 255])
                hist_g = np.histogram(tmp[1], bins=nb_bins, range=[0, 255])
                hist_b = np.histogram(tmp[2], bins=nb_bins, range=[0, 255])
                count_r += hist_r[0]
                count_g += hist_g[0]
                count_b += hist_b[0]


                #convert_tensor = transforms.ToTensor()
                #tmp = convert_tensor(tmp)
                #tmp = tmp.mul(255).clamp_(0, 255).permute(1, 2, 0).to('cpu').numpy()
                #tmp = tmp.astype(np.uint8)
                #t.append(skimage.measure.shannon_entropy(tmp))
                i = i + 1
                if(i==len(data_train)):
                    w=True
                    i=i-1
        #entropy.append(sum(t)/len(t))
        #cls.append(x)
        if(x == 99): break
        x = x + 1

    bins = hist_r[1]
    fig = plt.figure()
    plt.bar(bins[:-1], count_r, color='r', alpha=0.7)
    plt.bar(bins[:-1], count_g, color='g', alpha=0.7)
    plt.bar(bins[:-1], count_b, color='b', alpha=0.7)
    plt.savefig('min-entropy-histo.png')
    #arr = np.array(entropy)
    #print(arr)
    #idx = (-arr).argsort()[900:1000]
    #print(idx)
    #prit(arr[900:1000])





def main_x():
    nb_bins = 256
    count_r = np.zeros(nb_bins)
    count_g = np.zeros(nb_bins)
    count_b = np.zeros(nb_bins)


    for image in os.listdir('./test/'):
        img = Image.open('./test/'+image)
        x = np.array(img)
        x = x.transpose(2, 0, 1)
        hist_r = np.histogram(x[0], bins=nb_bins, range=[0, 255])
        hist_g = np.histogram(x[1], bins=nb_bins, range=[0, 255])
        hist_b = np.histogram(x[2], bins=nb_bins, range=[0, 255])
        count_r += hist_r[0]
        count_g += hist_g[0]
        count_b += hist_b[0]

    bins = hist_r[1]
    fig = plt.figure()
    plt.bar(bins[:-1], count_r, color='r', alpha=0.7)
    plt.bar(bins[:-1], count_g, color='g', alpha=0.7)
    plt.bar(bins[:-1], count_b, color='b', alpha=0.7)


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    cost = (end - start) / 86400
    payload = "Running Cost %.2f Days \n" % cost
    #logger.info(payload)


