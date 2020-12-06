from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os,time,cv2,scipy.io
import tensorflow as tf
import scipy.misc as sic
import subprocess
import numpy as np
from matplotlib.colors import hsv_to_rgb
from skimage.measure import compare_ssim, compare_psnr
from glob import glob
from losses import *

class polarization:
    # Fundamental property
    I_0 = None
    I_45 = None
    I_90 = None
    I_135 = None
    # Function
    def __init__(self,I_0,I_45,I_90,I_135):
        self.I_0 = I_0
        self.I_45 = I_45
        self.I_90 = I_90
        self.I_135 = I_135
        I = (self.I_0 + self.I_45 + self.I_90 + self.I_135)/ 2.
        Q = self.I_0 - self.I_90 
        U = self.I_45 - self.I_135
        Q[Q == 0] = 1e-10
        I[I == 0] = 1e-10
        self.rho = (np.sqrt(np.square(Q)+np.square(U))/I).clip(0,1)
        phi = 0.5*np.arctan(U/Q)
        cos_2phi = np.cos(2*phi)
        check_sign = cos_2phi * Q
        phi[check_sign<0] =  phi[check_sign<0] + math.pi/2.
        self.phi = (phi + math.pi)%math.pi
        self.Iun = 0.5*I

    def visualize_phi_hsv(self):
        phi = self.phi/math.pi
        h,w = phi.shape[:2]
        hsv = np.concatenate([phi[...,None], np.ones([h,w,1]),np.ones([h,w,1])],axis=2)
        self.phi_rgb = hsv_to_rgb(hsv)
        return  self.phi_rgb

    def visualize_rho_hsv(self):
        h,w = self.rho.shape[:2]
        hsv = np.concatenate([self.rho[...,None], np.ones([h,w,1]),np.ones([h,w,1])],axis=2)
        self.phi_rgb = hsv_to_rgb(hsv)
        return  self.phi_rgb
    
    def visualize_polarimg(self):
        h,w = self.rho.shape[:2]
        hsv = np.concatenate([phi[...,None]/math.pi, rho[...,None], np.ones([h,w,1])],axis=2)
        self.polarimg_rgb = hsv_to_rgb(hsv)
        return  self.polarimg_rgb


def tf_raw2data(raw):
    # Phi: AoLP, Rho: DoLP  
    phi, rho= tf_calculate_ADoLP(raw)
    Ip = tf.multiply(phi, raw[:,:,:,4:5])
    Inp = raw[:,:,:,4:5] - Ip
    return tf.concat([raw[:,:,:,:5], phi, rho, Ip, Inp, raw[:,:,:,5:]],axis=3)


def tf_overexp_mask(input):
    tmp = tf.reduce_max(input[:,:,:,:4],axis=3,keep_dims=True)
    zero_mat= tf.zeros(tf.shape(tmp),tf.float32)
    ones_mat= tf.ones(tf.shape(tmp),tf.float32)
    over_exp_mask = tf.where(tf.less(tmp,0.98),ones_mat,zero_mat) #if tmp larger than 0.98, over_exp_mask = 0
    over_exp_mask = tf.where(tf.greater(tmp,0.0001),over_exp_mask,zero_mat) # if tmp smaller than 0.02, over_exp_mask = 0
    return over_exp_mask


def get_metrics(metrics,out_mask, gt_target,gt_reflection,pred_image_t,pred_image_r):
    metrics["T_ssim"] += compare_ssim(0.5*gt_target[0,:,:,4]*out_mask[0,:,:,0], 0.5*pred_image_t[0,:,:,4]*out_mask[0,:,:,0])
    metrics["T_psnr"] += compare_psnr(0.5*gt_target[0,:,:,4]*out_mask[0,:,:,0], 0.5*pred_image_t[0,:,:,4]*out_mask[0,:,:,0], 1)
    metrics["R_ssim"] += compare_ssim(0.5*gt_reflection[0,:,:,4]*out_mask[0,:,:,0], 0.5*pred_image_r[0,:,:,4]*out_mask[0,:,:,0])
    metrics["R_psnr"] += compare_psnr(0.5*gt_reflection[0,:,:,4]*out_mask[0,:,:,0], 0.5*pred_image_r[0,:,:,4]*out_mask[0,:,:,0], 1)
    return metrics


def save_concat_img(out_mask, gt_input, gt_target,gt_reflection,pred_image_t,pred_image_r,save_path):
    out_img1= np.concatenate([2*gt_input[0,:,:,:3], 2*gt_target[0,:,:,:3],2*gt_reflection[0,:,:,:3]], axis=1)
    out_img2= np.concatenate([np.tile(2*out_mask[0,:,:,:],[1,1,3]),np.tile(pred_image_t[0,:,:,4:5],[1,1,3]),np.tile(pred_image_r[0,:,:,4:5],[1,1,3])], axis=1)
    out_img = np.concatenate([out_img1, out_img2], axis=0)
    out_img = np.power(np.minimum(np.maximum(0.5*out_img,0.0),1.0),1/2.2)*255.0
    #cv2.imwrite(save_path, np.uint8(out_img[::2,::2]))
    cv2.imwrite(save_path, np.uint8(out_img))
    return out_img


def save_results(all_loss_test, metrics, id, task,epoch):
    result=open("result/%s/score.txt"%task,'a')
    result.write("Epc: %03d Loss: %.5f | SSIM: %.3f PSNR: %.2f | SSIM: %.3f PSNR: %.2f \n"%\
        (epoch, np.mean(all_loss_test[np.where(all_loss_test)]), metrics["T_ssim"]/(id+1), metrics["T_psnr"]/(id+1), metrics["R_ssim"]/(id+1), metrics["R_psnr"]/(id+1)))
    result.close()


def crop_shape(tmp_all):
    h,w = tmp_all.shape[1:3]
    h = h//32*32
    w = w//32*32
    return h, w


def cnts_add_display(epoch, cnts, step, lossDict,st):
    cnts["cnt"]+=1
    step+=1
    cnts["all_r"] += lossDict["reflection"]
    cnts["all_t"] += lossDict["transmission"]
    cnts["all_pncc"] += lossDict["pncc"]
    cnts["all_recon"] += lossDict["reconstruct"]
    print("iter: %03d %03d || r:%.2f | t:%.2f | pncc:%.2f |recon:%.3f |time:%.2f"%\
        (epoch,cnts["cnt"], lossDict["reflection"], lossDict["transmission"], lossDict["pncc"], lossDict["reconstruct"],time.time()-st))
    return cnts, step


def raw2imgs(raw_img):
    I_90=raw_img[::2,::2] 
    I_45=raw_img[::2,1::2] 
    I_135=raw_img[1::2,::2] 
    I_0=raw_img[1::2,1::2]
    I = 0.5*(I_0 + I_45 + I_90 + I_135)
    return I_0, I_45, I_90, I_135, I 


def prepare_single_item(raw_img_name):
    raw_img = cv2.imread(raw_img_name, -1)/65535.
    I_0, I_45, I_90, I_135, I = raw2imgs(raw_img)
    return np.concatenate([i[np.newaxis,...,np.newaxis] for i in [I_0, I_45, I_90, I_135, I]],axis=3)


def prepare_single_MTR_item(MR_item):
    M_name, R_name = MR_item
    tf_M = prepare_single_item(M_name)
    tf_R = prepare_single_item(R_name)
    tf_T = (tf_M - tf_R).clip(0,2)
    return tf_M, tf_T, tf_R


def calculate_ADoLP(I1, I2, I3, I4, I):
    Q = I1 - I3 
    U = I2 - I4
    Q[Q == 0] = 0.0001
    I[I == 0] = 0.0001
    DoLP = np.sqrt(np.square(Q)+np.square(U))/I
    AoLP = 0.5*np.arctan(U/Q)
    # print(np.min(DoLP), np.max(DoLP))
#    AoLP = (AoLP + 0.786)/(2*0.786)
    DoLP[DoLP>1] = 1 
    return AoLP, DoLP


def crop_augmentation_MRT(im_M, im_T, im_R):
    h_orig,w_orig = im_R.shape[1:3]
    h_crop = h_orig//32*32
    w_crop = w_orig//32*32
    w_crop = 640 if w_crop > 640 else w_crop
    h_crop = 640 if h_crop > 640 else h_crop
    try:
        w_offset = 0 if w_crop==w_orig else np.random.randint(0, w_orig-w_crop)
        h_offset = 0 if h_crop==h_orig else np.random.randint(0, h_orig-h_crop)
    except:
        print("Original W %d, desired W %d"%(w_orig,w_crop))
        print("Original H %d, desired H %d"%(h_orig,h_crop))
    im_M = im_M[:,h_offset:h_offset+h_crop,w_offset:w_offset+w_crop,:]
    im_R = im_R[:,h_offset:h_offset+h_crop,w_offset:w_offset+w_crop,:]
    im_T = im_T[:,h_offset:h_offset+h_crop,w_offset:w_offset+w_crop,:]
    return im_M, im_T, im_R
 
def tf_calculate_ADoLP(I_all):
    I1, I2, I3, I4 = I_all[:,:,:,:1], I_all[:,:,:,1:2], I_all[:,:,:,2:3], I_all[:,:,:,3:4]
    I = 0.5 * (I1 + I2 + I3 + I4)+1e-4
    Q = I1 - I3 
    U = I2 - I4
    zero_mat = tf.zeros(tf.shape(I1), tf.float32)
    ones_mat = 1e-4 * tf.ones(tf.shape(I1), tf.float32)
    Q = tf.where(tf.equal(Q, zero_mat), ones_mat, Q)
    DoLP = tf.divide(tf.sqrt(tf.square(Q)+tf.square(U)), I)
    AoLP = 0.5*tf.atan(U/Q)
    #AoLP = (AoLP + 0.786)/(2*0.786)
    return AoLP, DoLP


def prepare_real_input(input, target, reflection, overexp_mask, ARGS):
    tf_target = target 
    tf_input = tf_raw2data(input)
    tf_reflection=tf_raw2data(reflection)
    tf_target=tf_raw2data(tf_target)
    real_input =tf.concat([tf_input,overexp_mask],axis=3)
    real_input = build_all_hyper(real_input)
    return tf_input, tf_reflection, tf_target, real_input
