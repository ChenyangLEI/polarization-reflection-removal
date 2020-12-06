from __future__ import division
import os,time,cv2,scipy.io
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
import subprocess
import utils
from network import DialUNet as UNet
from glob import glob
import losses as loss
import random

parser = argparse.ArgumentParser()
parser.add_argument("--loss", default="Loss_perceptual", help="choose the loss type")
parser.add_argument("--is_pol", default=1,type=int, help="choose the loss type")
parser.add_argument("--task", default="CVPR2020",help="path to folder containing the model")
parser.add_argument("--debug", default=0, type=int, help="DEBUG or not")
parser.add_argument("--save_model_freq", default=5, type=int, help="frequency to save model")

ARGS = parser.parse_args()
DEBUG = ARGS.debug
save_model_freq = ARGS.save_model_freq 
task=ARGS.task
print(ARGS)

seed = 2020
np.random.seed(seed)
tf.set_random_seed(seed)
random.seed(seed)


continue_training=True
os.environ["CUDA_VISIBLE_DEVICES"]=str(np.argmax( [int(x.split()[2]) for x in subprocess.Popen("nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True, stdout=subprocess.PIPE).stdout.readlines()]))


#--------------Prepare Data----------------------#
base_folder = "./polar_rr_dataset/"
train_M_names = sorted(glob("{}/train/Mixed/*png".format(base_folder)))
train_R_names = sorted(glob("{}/train/Reflection/*png".format(base_folder)))
train_T_names = sorted(glob("{}/train/Transmission/*png".format(base_folder)))

val_M_names = sorted(glob("{}/val/Mixed/*png".format(base_folder)))
val_R_names = sorted(glob("{}/val/Reflection/*png".format(base_folder)))
val_T_names = sorted(glob("{}/val/Transmission/*png".format(base_folder)))

assert len(train_M_names) == len(train_R_names), "Number of Mixed images and Reflection images are unequal in TRAIN set"
assert len(train_M_names) == len(train_R_names), "Number of Mixed images and Reflection images are unequal in VAL set"

num_train, num_val = len(train_M_names), len(val_M_names)
print('Data load succeed!', num_train, num_val)

#--------------Prepare Data----------------------#


print("debug:", DEBUG)
if DEBUG:
    task = 'DEBUG' + task
    num_train = 10
    num_val = 2
    save_model_freq = 1 

# set up the model and define the graph
with tf.variable_scope(tf.get_variable_scope()):
    input=tf.placeholder(tf.float32,shape=[None,None,None,5])
    reflection=tf.placeholder(tf.float32,shape=[None,None,None,5])
    target=tf.placeholder(tf.float32,shape=[None,None,None,5])
    overexp_mask = utils.tf_overexp_mask(input)
    tf_input, tf_reflection, tf_target, real_input = utils.prepare_real_input(input, target, reflection, overexp_mask, ARGS)
    reflection_layer=UNet(real_input, ext='Ref_') #real_reflect = build_one_hyper(reflection_layer[...,4:5])
    transmission_layer=UNet(tf.concat([real_input, reflection_layer],axis=3),ext='Tran_') 
    lossDict = {}

    lossDict["percep_t"]=0.2*loss.compute_percep_loss(0.5 * tf_target[...,4:5],  0.5*transmission_layer[...,4:5], overexp_mask, reuse=False )
    lossDict["percep_r"]=0.2*loss.compute_percep_loss(0.5 * tf_reflection[...,4:5], 0.5*reflection_layer[...,4:5], overexp_mask, reuse=True)

    lossDict["pncc"] = 6*loss.compute_percep_ncc_loss(tf.multiply(0.5*transmission_layer[...,4:5],overexp_mask), 
        tf.multiply(0.5*reflection_layer[...,4:5],overexp_mask))

    lossDict["reconstruct"]= loss.mask_reconstruct_loss(tf_input[...,4:5], transmission_layer[...,4:5], reflection_layer[...,4:5], overexp_mask)
    
    lossDict["reflection"] = lossDict["percep_r"]
    lossDict["transmission"]=lossDict["percep_t"]
    lossDict["all_loss"] = lossDict["reflection"] + lossDict["transmission"] + lossDict["pncc"]


######### Session #########
all_vars=[var for var in tf.trainable_variables() if 'g_' in var.name]
all_opt=tf.train.AdamOptimizer(learning_rate=0.0001).minimize(lossDict["all_loss"],var_list=all_vars)
for var in tf.trainable_variables():
    print("Listing trainable variables ... ",var)

saver=tf.train.Saver(max_to_keep=20)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)
sess.run(tf.global_variables_initializer())
var_restore = [v for v in tf.trainable_variables()]
saver_restore=tf.train.Saver(var_restore)
ckpt=tf.train.get_checkpoint_state('result/'+task)
######### Session #########

print("[i] contain checkpoint: ", ckpt)
if ckpt and continue_training:
    saver_restore=tf.train.Saver([var for var in tf.trainable_variables()])
    print('loaded '+ckpt.model_checkpoint_path)
    saver_restore.restore(sess,ckpt.model_checkpoint_path)


maxepoch=201
step = 0
for epoch in range(1,maxepoch):
    print("Processing epoch %d"%epoch)
    if os.path.isdir("result/%s/%04d"%(task,epoch)):
        continue
    else:
        os.makedirs("result/%s/%04d"%(task,epoch))
    cnts = {"cnt":0, "all_t":0, "all_r":0, "all_pncc":0,"all_recon":0}
    

    for id in np.random.permutation(num_train):
        tmp_M = utils.prepare_single_item(train_M_names[id])
        tmp_R = utils.prepare_single_item(train_R_names[id])
        tmp_T = utils.prepare_single_item(train_T_names[id])
        tmp_M, tmp_T, tmp_R = utils.crop_augmentation_MRT(tmp_M, tmp_T, tmp_R)
        fetch_list=[all_opt, overexp_mask, transmission_layer, reflection_layer, tf_input, tf_target, tf_reflection, lossDict]
        st=time.time()
        h,w=utils.crop_shape(tmp_M)
        magic = np.random.random()
        tmp_M = tmp_M[:,:h,:w,:]
        tmp_R = tmp_R[:,:h,:w,:]
        tmp_T = tmp_T[:,:h,:w,:]

        if magic < 0.5:
            tmp_M = tmp_M[:,::2,::2,:]
            tmp_R = tmp_R[:,::2,::2,:]
            tmp_T = tmp_T[:,::2,::2,:]


        _,out_mask, pred_image_t,pred_image_r,gt_input,gt_target,gt_reflection,crt_lossDict=sess.run(fetch_list,
                feed_dict={input:tmp_M, reflection:tmp_R, target:tmp_T})

        cnts,step=utils.cnts_add_display(epoch,cnts,step,crt_lossDict,st)
        if ((id % 20) == 0 and (epoch % save_model_freq == 0) ) or (step % 100 == 1) :
            utils.save_concat_img(out_mask, gt_input, gt_target,gt_reflection,pred_image_t,pred_image_r, "result/%s/%04d/train_%06d.jpg"%(task, epoch, id))


    # save model and images every epoch
    if epoch % save_model_freq == 0:
        all_loss_test=np.zeros(num_val, dtype=float)#num_val*num_val//2, dtype=float)
        metrics = {"T_ssim":0,"T_psnr":0,"R_ssim":0, "R_psnr":0}
        saver.save(sess,"result/%s/model.ckpt"%task)
        saver.save(sess,"result/%s/%04d/model.ckpt"%(task,epoch))
        for id in range(num_val):
            tmp_M = utils.prepare_single_item(val_M_names[id])
            tmp_R = utils.prepare_single_item(val_R_names[id])
            tmp_T = utils.prepare_single_item(val_T_names[id])

            h, w = utils.crop_shape(tmp_M)
            out_loss,out_mask,pred_image_t, pred_image_r, gt_input,gt_target,gt_reflection=sess.run([lossDict["transmission"], 
                overexp_mask, transmission_layer, reflection_layer,tf_input,tf_target,tf_reflection],
                feed_dict={input:tmp_M[:,:h:2,:w:2,:], reflection:tmp_R[:,:h:2,:w:2,:], target:tmp_T[:,:h:2,:w:2,:]})
            print("Epc: %3d, shape of outputs: "%epoch,pred_image_t.shape, pred_image_r.shape)
            utils.save_concat_img(out_mask,gt_input, gt_target,gt_reflection,pred_image_t,pred_image_r, "result/%s/%04d/val_%06d.jpg"%(task, epoch, id))
            all_loss_test[id]=out_loss
            metrics = utils.get_metrics(metrics,out_mask, gt_target,gt_reflection,pred_image_t,pred_image_r)
        utils.save_results(all_loss_test, metrics, id, task,epoch)
