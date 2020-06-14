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
#from losses import *
def build_net(ntype,nin,nwb=None,name=None):
    if ntype=='conv':
        return tf.nn.relu(tf.nn.conv2d(nin,nwb[0],strides=[1,1,1,1],padding='SAME',name=name)+nwb[1])
    elif ntype=='pool':
        return tf.nn.avg_pool(nin,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def get_weight_bias(vgg_layers,i):
    weights=vgg_layers[i][0][0][2][0][0]
    weights=tf.constant(weights)
    bias=vgg_layers[i][0][0][2][0][1]
    bias=tf.constant(np.reshape(bias,(bias.size)))
    return weights,bias
    
def lrelu(x):
    return tf.maximum(x*0.2,x)

def relu(x):
    return tf.maximum(0.0,x)

def identity_initializer():
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        array = np.zeros(shape, dtype=float)
        cx, cy = shape[0]//2, shape[1]//2
        for i in range(np.minimum(shape[2],shape[3])):
            array[cx, cy, i, i] = 1
        return tf.constant(array, dtype=dtype)
    return _initializer

def nm(x):
    w0=tf.Variable(1.0,name='w0')
    w1=tf.Variable(0.0,name='w1')
    return w0*x+w1*slim.batch_norm(x)

vgg_path=scipy.io.loadmat('./VGG_Model/imagenet-vgg-verydeep-19.mat')
print("[i] Loaded pre-trained vgg19 parameters")
# build VGG19 to load pre-trained parameters
def build_vgg19(input,reuse=False):
    with tf.variable_scope("vgg19"):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        net={}
        vgg_layers=vgg_path['layers'][0]
        net['input']=input-np.array([123.6800, 116.7790, 103.9390]).reshape((1,1,1,3))
        net['conv1_1']=build_net('conv',net['input'],get_weight_bias(vgg_layers,0),name='vgg_conv1_1')
        net['conv1_2']=build_net('conv',net['conv1_1'],get_weight_bias(vgg_layers,2),name='vgg_conv1_2')
        net['pool1']=build_net('pool',net['conv1_2'])
        net['conv2_1']=build_net('conv',net['pool1'],get_weight_bias(vgg_layers,5),name='vgg_conv2_1')
        net['conv2_2']=build_net('conv',net['conv2_1'],get_weight_bias(vgg_layers,7),name='vgg_conv2_2')
        net['pool2']=build_net('pool',net['conv2_2'])
        net['conv3_1']=build_net('conv',net['pool2'],get_weight_bias(vgg_layers,10),name='vgg_conv3_1')
        net['conv3_2']=build_net('conv',net['conv3_1'],get_weight_bias(vgg_layers,12),name='vgg_conv3_2')
        net['conv3_3']=build_net('conv',net['conv3_2'],get_weight_bias(vgg_layers,14),name='vgg_conv3_3')
        net['conv3_4']=build_net('conv',net['conv3_3'],get_weight_bias(vgg_layers,16),name='vgg_conv3_4')
        net['pool3']=build_net('pool',net['conv3_4'])
        net['conv4_1']=build_net('conv',net['pool3'],get_weight_bias(vgg_layers,19),name='vgg_conv4_1')
        net['conv4_2']=build_net('conv',net['conv4_1'],get_weight_bias(vgg_layers,21),name='vgg_conv4_2')
        net['conv4_3']=build_net('conv',net['conv4_2'],get_weight_bias(vgg_layers,23),name='vgg_conv4_3')
        net['conv4_4']=build_net('conv',net['conv4_3'],get_weight_bias(vgg_layers,25),name='vgg_conv4_4')
        net['pool4']=build_net('pool',net['conv4_4'])
        net['conv5_1']=build_net('conv',net['pool4'],get_weight_bias(vgg_layers,28),name='vgg_conv5_1')
        net['conv5_2']=build_net('conv',net['conv5_1'],get_weight_bias(vgg_layers,30),name='vgg_conv5_2')
        return net

# our reflection removal model
def build_hyper(input):
    print("[i] Hypercolumn ON, building hypercolumn features ... ")
    vgg19_features=build_vgg19(tf.sqrt(tf.tile(0.5*input[:,:,:,4:5],[1,1,1,3]))*255.0)
    for layer_id in range(1,2):
        if layer_id == 1:
	        vgg19_f = vgg19_features['conv%d_1'%layer_id]
	        input = tf.concat([tf.image.resize_bilinear(vgg19_f,(tf.shape(input)[1],tf.shape(input)[2]))/255.0,input], axis=3)
        vgg19_f = vgg19_features['conv%d_2'%layer_id]
        input = tf.concat([tf.image.resize_bilinear(vgg19_f,(tf.shape(input)[1],tf.shape(input)[2]))/255.0,input], axis=3)
    return input

def build_one_hyper(input):
    print("[i] Hypercolumn ON, building hypercolumn features ... ")
    zero_mat= tf.zeros(tf.shape(input),tf.float32)
    input  = tf.where(tf.greater(input,0),input,zero_mat)
 
    vgg19_features=build_vgg19(tf.pow(tf.tile(input[:,:,:,0:1],[1,1,1,3]),1/2.2)*255.0)
    for layer_id in range(1,2):
        if layer_id == 1:
            vgg19_f = vgg19_features['conv%d_1'%layer_id]
            input = tf.concat([tf.image.resize_bilinear(vgg19_f,(tf.shape(input)[1],tf.shape(input)[2]))/255.0,input], axis=3)
        vgg19_f = vgg19_features['conv%d_2'%layer_id]
        input = tf.concat([tf.image.resize_bilinear(vgg19_f,(tf.shape(input)[1],tf.shape(input)[2]))/255.0,input], axis=3)
    return input

def build_all_hyper(input):
    print("[i] Hypercolumn ON, building hypercolumn features ... ")
    all_hypers = [input]
    for i in range(5):
        input_hyper = build_one_hyper(input[:,:,:,i:i+1]) if i < 4 else build_one_hyper(0.5*input[:,:,:,i:i+1]) 
        all_hypers.append(input_hyper)
    return tf.concat(all_hypers,axis=3)
 
def build(input):
    net=slim.conv2d(input,channel,[1,1],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv0')
    net=slim.conv2d(net,channel,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv1')
    net=slim.conv2d(net,channel,[3,3],rate=2,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv2')
    net=slim.conv2d(net,channel,[3,3],rate=4,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv3')
    net=slim.conv2d(net,channel,[3,3],rate=8,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv4')
    net=slim.conv2d(net,channel,[3,3],rate=16,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv5')
    net=slim.conv2d(net,channel,[3,3],rate=32,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv6')
    net=slim.conv2d(net,channel,[3,3],rate=64,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv7')
    net=slim.conv2d(net,channel,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv9')
    net=slim.conv2d(net, 5 * 2,[1,1],rate=1,activation_fn=None,scope='g_conv_last') # output 6 channels --> 3 for transmission layer and 3 for reflection layer
    return net

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


def prepare_final_MR(train_Mpath= '../data_new/Data_Polar_Clean/MTR/train/', ):
    train_items,val_items,test_items=[],[],[]
    imgs = glob(data_path + "/*npy")
    imgs.sort()
    for idx in range(len(imgs)//3):
        train_items.append(imgs[3*idx:3*idx+3])

    imgs = glob(data_path.replace("train","val") + "/*npy")
    imgs.sort()
    for idx in range(len(imgs)//3):
        val_items.append(imgs[3*idx:3*idx+3])

    return train_items, val_items, test_items


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


def visualize_phi_and_rho(phi, rho):
    phi = phi[...,np.newaxis]
    rho = rho[...,np.newaxis]
    h,w = phi.shape[:2]
    hsv = np.concatenate([phi, rho, np.ones([h,w,1])],axis=2)
    rgb = hsv_to_rgb(hsv)
    return   rgb


def visualize_phi(phi):
    phi = phi[:,:,np.newaxis]
    h,w = phi.shape[:2]
    hsv = np.concatenate([phi, np.ones([h,w,1]), np.ones([h,w,1])],axis=2)
    rgb = hsv_to_rgb(hsv)
    return  rgb


def crop_augmentation_MRT(im_M, im_T, im_R):
    h_orig,w_orig = im_R.shape[1:3]
    h_crop = h_orig//32*32
    w_crop = w_orig//32*32
    w_crop = 640 if w_crop > 640 else w_crop
    h_crop = 640 if h_crop > 640 else h_crop
    try:
        w_offset = np.random.randint(0, w_orig-w_crop)
        h_offset = np.random.randint(0, h_orig-h_crop)
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
