# Visual-Complexity-and-Unlearnable-Examples
This is a repository which contains the scripts used for my Master thesis **Visual Complexity and Unlearnable Examples**. This research extends the work of Huang et al. [https://arxiv.org/abs/2101.04898] and Liu et al. [https://arxiv.org/abs/2111.13244] for creating Unlearnable examples for visually complex images. Visual complexity  is defined by the number of objects and classes present in an image,  thireir positioning, the incongruous
colors present in an image and th resolution. 
Unlearnable examples are images that have been perturbed by imperceptible error-mininimizing noise that make the training set "unuseable" or "un-learnable" for a Deep Neural Network. Initial work by both  Huang et al. and Liu et al. have shown promising results however more visually complex datasets have to be explored to further understand how exactly Unlearnable examples work.
## ImageNet-100-Pytorch-main 
Directory to generate the subsets of different levels of visual complexity. The subsets are created from the ILSVRC-2012 dataset. To run this you need to have downloaded this [https://image-net.org/challenges/LSVRC/2012/2012-downloads.php]. 
### Generate the subsets
Run the job.sh code to generate datasets of different visual complexity. In the folder tmp you can find the files min100.txt (100 visually simple classes), random.txt (100 classes of unknown complexity) and max100.txt (100 visually complex classes). The .txt files for the min100.txt and max100.txt have been created by calculating the entropy of each class in the ILSVRC-2012 dataset. The 100 classes with the highest entropy ( most visually complex ) were copied into the max100.txt file and the 100 classes with the lowest entropy ( most visually simple ) were copied into the min100.txt
The arguments for the batch script  are:

arguments:
  - `--source_folder`: specify the ILSVRC-2012 data folder (e.g., `~/ILSVRC2012/train`)
  - `--target_folder`: specify the ImageNet-100 data folder (e.g., `~/Imagenet-100/train`)
  - `--target_class`: specify the ImageNet-100 txt file of different levels of visual complexity [ 'min100.txt','max100.txt','random.txt']
 An example of how to generate the train data for 100 visually simple classes is given as :
 ```
 python generate_IN100.py \
        --source_folder ~/ILSVRC2012/train\
        --target_folder ~/Imagenet-100/train\
        --target_class tmp/min100.txt

 ```
 An example of how to generate the val data for 100 visually simple classes is given as :
 ```
 python generate_IN100.py \
        --source_folder ~/ILSVRC2012/val\
        --target_folder ~/Imagenet-100/val\
        --target_class tmp/min100.txt
 ```
 
## Citation

```
@article{DBLP:journals/corr/abs-2111-13244,
  author    = {Zhuoran Liu and
               Zhengyu Zhao and
               Alex Kolmus and
               Tijn Berns and
               Twan van Laarhoven and
               Tom Heskes and
               Martha A. Larson},
  title     = {Going Grayscale: The Road to Understanding and Improving Unlearnable
               Examples},
  journal   = {CoRR},
  volume    = {abs/2111.13244},
  year      = {2021},
  url       = {https://arxiv.org/abs/2111.13244},
  eprinttype = {arXiv},
  eprint    = {2111.13244},
  timestamp = {Wed, 01 Dec 2021 15:16:43 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2111-13244.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}

@article{DBLP:journals/corr/abs-2101-04898,
  author    = {Hanxun Huang and
               Xingjun Ma and
               Sarah Monazam Erfani and
               James Bailey and
               Yisen Wang},
  title     = {Unlearnable Examples: Making Personal Data Unexploitable},
  journal   = {CoRR},
  volume    = {abs/2101.04898},
  year      = {2021},
  url       = {https://arxiv.org/abs/2101.04898},
  eprinttype = {arXiv},
  eprint    = {2101.04898},
  timestamp = {Fri, 22 Jan 2021 15:16:00 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2101-04898.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
