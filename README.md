# CVPR2020-Polarized-Reflection-Removal-with-Perfect-Alignment
Code for CVPR2020 paper "Polarized Reflection Removal with Perfect Alignment in the Wild"

[paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Lei_Polarized_Reflection_Removal_With_Perfect_Alignment_in_the_Wild_CVPR_2020_paper.pdf)
|[project website](https://leichenyang.weebly.com/project-polarized.html)


## To do list
- [x] Dependencey
- [x] Inference code
- [x] Dataset
- [] Training code

## PolarRR Dataset
Please download the dataset on [Google Drive](https://drive.google.com/file/d/1iRHs23jSOfim-toWXqwJJ3CaTWUHSaYF/view?usp=sharing)

## Setup

### Environment
This code is based on tensorflow. It has been tested on Ubuntu 18.04 LTS.

Anaconda is recommended: [Ubuntu 18.04](https://www.digitalocean.com/community/tutorials/how-to-install-the-anaconda-python-distribution-on-ubuntu-18-04)
| [Ubuntu 16.04](https://www.digitalocean.com/community/tutorials/how-to-install-the-anaconda-python-distribution-on-ubuntu-16-04)

After installing Anaconda, you can setup the environment simply by

```
conda env create -f environment.yml
```

### Download checkpoint and VGG model

Download the ckpt and VGG model from the Google drive, put them in the correct path:

VGG_Model/imagenet-vgg-verydeep-19.mat

ckpt/Submission_ckpt/checkpoint


## Inference
```
python final_infer.py --task Submission_ckpt --test_dir demo
```

The results are placed in ./test_result


## Citation
If you find this work useful for your research, please cite:
```
@InProceedings{Lei_2020_CVPR,
author = {Lei, Chenyang and Huang, Xuhua and Zhang, Mengdi and Yan, Qiong and Sun, Wenxiu and Chen, Qifeng},
title = {Polarized Reflection Removal With Perfect Alignment in the Wild},
booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```

## Contact
Please contact me if there is any question (Chenyang Lei, leichenyang7@gmail.com)
