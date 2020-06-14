from __future__ import division
import os,time,cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import utils
from network import DialUNet as UNet
from glob import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="pre-trained",help="path to folder containing the model")
parser.add_argument("--test_dir", default="../data/Submission_MMR/png_raw/test/Mixed", help="path to test folder")
parser.add_argument("--output_dir", default="../result/", help="path to test folder")
parser.add_argument("--is_pol", default=1,type=int, help="choose the loss type")
parser.add_argument("--use_gpu", default=0,type=int, help="choose the loss type")
ARGS = parser.parse_args()
task=ARGS.task
print(ARGS)
if ARGS.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"]=str(np.argmax( [int(x.split()[2]) for x in subprocess.Popen("nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True, stdout=subprocess.PIPE).stdout.readlines()]))
else:    
    os.environ["CUDA_VISIBLE_DEVICES"]=''


test_names= sorted(glob(ARGS.test_dir + "/*png"))
print('Data load succeed!')

# set up the model and define the graph
with tf.variable_scope(tf.get_variable_scope()):
    input=tf.placeholder(tf.float32,shape=[None,None,None,5])
    reflection=tf.placeholder(tf.float32,shape=[None,None,None,5])
    target=tf.placeholder(tf.float32,shape=[None,None,None,5])
    overexp_mask = utils.tf_overexp_mask(input)
    tf_input, tf_reflection, tf_target, real_input = utils.prepare_real_input(input, target, reflection, overexp_mask, ARGS)
    reflection_layer=UNet(real_input, ext='Ref_')
    transmission_layer = UNet(tf.concat([real_input, reflection_layer],axis=3),ext='Tran_') 
 

######### Session #########
saver=tf.train.Saver(max_to_keep=10)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)
sess.run(tf.global_variables_initializer())
var_restore = [v for v in tf.trainable_variables()]
saver_restore=tf.train.Saver(var_restore)
for var in tf.trainable_variables():
    print("Listing trainable variables ... ")
    print(var)


ckpt=tf.train.get_checkpoint_state('../result/'+task)
print("[i] contain checkpoint: ", ckpt)
if ckpt:
    saver_restore=tf.train.Saver([var for var in tf.trainable_variables()])
    print('loaded '+ckpt.model_checkpoint_path)
    saver_restore.restore(sess,ckpt.model_checkpoint_path)
else:
    print("There is no checkpoint.")

if not os.path.isdir("test_result/{}".format(task)):  
    os.makedirs("test_result/{}/png_raw".format(task))
    os.makedirs("test_result/{}/jpg_gamma".format(task))

cnt = 0
for id in range(len(test_names)):
    st=time.time()
    item = test_names[id]
    tmp_all = utils.prepare_single_item(item)
    h,w = utils.crop_shape(tmp_all)
    out_mask, pred_image_t, pred_image_r, gt_input=sess.run(\
        [overexp_mask,transmission_layer, reflection_layer,tf_input],feed_dict={input:tmp_all[:,:h,:w,:]}) 
    print("output dir:{}, shape of outputs: ".format("test_result/" + task), pred_image_t.shape, pred_image_r.shape, np.mean(pred_image_r), np.mean(pred_image_t))
    cv2.imwrite("test_result/{}/png_raw/{}".format(task,item.split("/")[-1][:-4]+"_1t.png"), np.uint16((0.5*pred_image_t[0,:,:,4]).clip(0,1)*65535.0))
    cv2.imwrite("test_result/{}/png_raw/{}".format(task,item.split("/")[-1][:-4]+"_2r.png"), np.uint16((0.5*pred_image_r[0,:,:,4]).clip(0,1)*65535.0))
    cv2.imwrite("test_result/{}/png_raw/{}".format(task,item.split("/")[-1][:-4]+"_0m.png"), np.uint16((0.5*tmp_all[0,:h,:w,4]).clip(0,1)*65535.0))
    cv2.imwrite("test_result/{}/jpg_gamma/{}".format(task,item.split("/")[-1][:-4]+"_1t.jpg"), np.uint8(np.power((0.5*pred_image_t[0,:,:,4]).clip(0,1),1/2.2)*255.0))
    cv2.imwrite("test_result/{}/jpg_gamma/{}".format(task,item.split("/")[-1][:-4]+"_2r.jpg"), np.uint8(np.power((0.5*pred_image_r[0,:,:,4]).clip(0,1),1/2.2)*255.0))
    cv2.imwrite("test_result/{}/jpg_gamma/{}".format(task,item.split("/")[-1][:-4]+"_0m.jpg"), np.uint8(np.power((0.5*tmp_all[0,:h,:w,4]).clip(0,1),1/2.2)*255.0))
print(cnt)   
