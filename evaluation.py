from imageio import imread, imsave
from glob import glob
from skimage.measure import compare_psnr, compare_ssim
import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--pred_T_dir", default="./test_result/Submission_ckpt/png_raw/", help="path to folder containing the transmission results")
parser.add_argument("--pred_R_dir", default="./test_result/Submission_ckpt/png_raw/", help="path to folder containing the reflection results")
parser.add_argument("--test_dir", default="./polar_rr_dataset/test/Transmission", help="path to test folder")

ARGS = parser.parse_args()
print(ARGS)

GT_names = sorted(glob("{}/*png".format(ARGS.test_dir)))

def ours_results(path="./test_result/Submission_ckpt/png_raw/*t.png"):
    return sorted(glob(path))

def prepare_single_item(raw_img_name):
    raw_img = cv2.imread(raw_img_name, -1)/65535.
    I_0, I_45, I_90, I_135, I = raw2imgs(raw_img)
    return np.concatenate([i[np.newaxis,...,np.newaxis] for i in [I_0, I_45, I_90, I_135, I]],axis=3)

def raw2imgs(raw_img):
    I_90=raw_img[::2,::2] 
    I_45=raw_img[::2,1::2] 
    I_135=raw_img[1::2,::2] 
    I_0=raw_img[1::2,1::2]
    I = 0.5*(I_0 + I_45 + I_90 + I_135)
    return I_0, I_45, I_90, I_135, I 



def compare_psnr_ssim(img1, img2, input=None):
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    h, w = min(h1,h2), min(w1, w2)
    h = h // 32 * 32
    w = w // 32 * 32
    img1 = img1[:h,:w]
    img2 = img2[:h,:w]
    print(img1.shape, img2.shape,np.mean(img1), np.mean(img2))    
    ssim = compare_ssim(img1, img2)
    psnr = compare_psnr(img1, img2,1)
    return ssim, psnr


all_ssim, all_psnr,cnt = 0,0,0
pred_images =ours_results(path=ARGS.pred_T_dir + "*t.png") 
print(len(pred_images),len(GT_names))
assert len(pred_images) == len(GT_names), "Folders of test_pred are not same! pred: {}, gt: {}".format(len(pred_images), len(GT_names))
for idx in range(len(pred_images)):
    pred_T = cv2.imread(pred_images[idx],-1)/65535.
    T = prepare_single_item(GT_names[idx])
    T = 0.5 * T[0,...,4]
    print(GT_names[idx], pred_images[idx])
    ssim, psnr=  compare_psnr_ssim(pred_T, T)
    all_ssim += ssim
    all_psnr += psnr
    cnt += 1
    print("{:03d} {:3f} {:2f}".format(cnt, ssim, psnr))
print(cnt, all_ssim*1.0/(cnt+1), all_psnr*1.0/(cnt+1))

