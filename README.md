# CVPR2020-Polarized-Reflection-Removal-with-Perfect-Alignment
Code for CVPR2020 paper "Polarized Reflection Removal with Perfect Alignment in the Wild"
## To do list
- [] Dependencey
- [] Inference code
- [] Training code
- [] Dataset

## Environment
The code is tested on Ubuntu 18.04, Cuda 9.1.

Anaconda is recommended: 

    Ubuntu 18.04: https://www.digitalocean.com/community/tutorials/how-to-install-the-anaconda-python-distribution-on-ubuntu-18-04

    Ubuntu 16.04: https://www.digitalocean.com/community/tutorials/how-to-install-the-anaconda-python-distribution-on-ubuntu-16-04

After installing Anaconda, you can setup the environment simply by

```
conda env create -f environment.yml
```

## Download checkpoint and VGG model

Download the ckpt and VGG model from the Google drive, put them in the correct path:

VGG_Model/imagenet-vgg-verydeep-19.mat

ckpt/Submission_ckpt/checkpoint


## Inference
```
python final_infer.py --task Submission_ckpt --test_dir demo
```

The results are placed in test_result
