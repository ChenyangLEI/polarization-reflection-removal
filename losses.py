from __future__ import division
import os
import time
import cv2
import scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import argparse
import subprocess
import utils
from network import DialUNet as UNet
from glob import glob


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

# functions to compute different loss terms

def mask_reconstruct_loss(input, out1, out2, valid_mask):
	output = out1 + out2
	return tf.reduce_mean(tf.multiply(tf.abs(input - output),valid_mask))

def compute_edge(img):
    edge=tf.abs(img[:,:-2,1:-1,:]-img[:,1:-1,1:-1,:]) + tf.abs(img[:,2:,1:-1,:]-img[:,1:-1,1:-1,:])+\
        tf.abs(img[:,1:-1,:-2,:]-img[:,1:-1,1:-1,:])+tf.abs(img[:,1:-1,2:,:]-img[:,1:-1,1:-1,:]) 
    return edge

def compute_edge_loss(img1, img2, mask):
    edge1 = compute_edge(img1)
    edge2 = compute_edge(img2)
    loss = []
    loss.append(tf.reduce_mean(tf.abs(edge1 - edge2)[:,:,:,0]*mask[:,1:-1,1:-1,0]))
    for l in range(4):
        img1=tf.nn.avg_pool(img1, [1,2,2,1], [1,2,2,1], padding='SAME')
        img2=tf.nn.avg_pool(img2, [1,2,2,1], [1,2,2,1], padding='SAME')
        mask=tf.nn.avg_pool(mask, [1,2,2,1], [1,2,2,1], padding='SAME')
        edge1 = compute_edge(img1)
        edge2 = compute_edge(img2)
        loss.append(tf.reduce_mean(tf.abs(edge1 - edge2)[:,:,:,0]*mask[:,1:-1,1:-1,0]))
    return sum(loss)/4.0


def compute_reconstruct_loss(tmp_M, tmp_R, tmp_mask, mask):
    I_M, I_R, I_T = tmp_M[:,:,:,0], tmp_R[:,:,:,0],tmp_mask[:,:,:,0]
    DoLP_M, DoLP_R, DoLP_T = tmp_M[:,:,:,1], tmp_R[:,:,:,1],tmp_mask[:,:,:,1]
    AoLP_M, AoLP_R, AoLP_T = tmp_M[:,:,:,2], tmp_R[:,:,:,2],tmp_mask[:,:,:,2]


    I_np_loss= tf.reduce_mean(mask*tf.abs(I_M*(-DoLP_M+1) - (I_R*(-DoLP_R+1)+I_T*(-DoLP_T+1))))
    Ip_losses = []
    for i in range(4):
        angle = 3.1415926*0.25*i
        Ip_M = I_M*DoLP_M*tf.cos(AoLP_M-angle)*tf.cos(AoLP_M-angle)
        Ip_R = I_R*DoLP_R*tf.cos(AoLP_R-angle)*tf.cos(AoLP_R-angle)
        Ip_T = I_T*DoLP_T*tf.cos(AoLP_T-angle)*tf.cos(AoLP_T-angle)
        Ip_losses.append(tf.reduce_mean(tf.abs(Ip_M - Ip_R - Ip_T)))
    
    I_p_loss=sum(Ip_losses)

    return I_np_loss, I_p_loss

def compute_Ip_loss(input, output, mask):
    Ip_AoLP = tf.reduce_mean(mask*tf.sqrt(input[:,:,:,1])*tf.exp(-tf.cos(input[:,:,:,2]-output[:,:,:,2])))
    Ip_DoLP=  tf.reduce_mean(mask*tf.square(input[:,:,:,1] - output[:,:,:,1]))
    return Ip_AoLP, Ip_DoLP

def compute_percep_loss(input, output, mask, reuse=False):
    input = tf.tile(tf.multiply(input,mask), [1,1,1,3])
    output= tf.tile(tf.multiply(output,mask),[1,1,1,3])

    vgg_real=build_vgg19(output*255.0,reuse=reuse)
    vgg_fake=build_vgg19(input*255.0,reuse=True)
    p0=tf.reduce_mean(tf.abs(vgg_real['input'] - vgg_fake['input']))
    p1_1=tf.reduce_mean(tf.abs(vgg_real['conv1_1']-vgg_fake['conv1_1']))
    p1=tf.reduce_mean(tf.abs(vgg_real['conv1_2']-vgg_fake['conv1_2']))/2.6
    p2=tf.reduce_mean(tf.abs(vgg_real['conv2_2']-vgg_fake['conv2_2']))/4.8
    p3=tf.reduce_mean(tf.abs(vgg_real['conv3_2']-vgg_fake['conv3_2']))/3.7
    p4=tf.reduce_mean(tf.abs(vgg_real['conv4_2']-vgg_fake['conv4_2']))/5.6
    p5=tf.reduce_mean(tf.abs(vgg_real['conv5_2']-vgg_fake['conv5_2']))*10/1.5

    loss = (p0+p1+p1_1+p2+p3+p4+p5)
    return loss 

def compute_exclusion_loss(img1,img2,level=1):
    gradx_loss=[]
    grady_loss=[]
    
    for l in range(level):
        gradx1, grady1=compute_gradient(img1)
        gradx2, grady2=compute_gradient(img2)
        alphax=2.0*tf.reduce_mean(tf.abs(gradx1))/tf.reduce_mean(tf.abs(gradx2))
        alphay=2.0*tf.reduce_mean(tf.abs(grady1))/tf.reduce_mean(tf.abs(grady2))
        
        gradx1_s=(tf.nn.sigmoid(gradx1)*2)-1
        grady1_s=(tf.nn.sigmoid(grady1)*2)-1
        gradx2_s=(tf.nn.sigmoid(gradx2*alphax)*2)-1
        grady2_s=(tf.nn.sigmoid(grady2*alphay)*2)-1

        gradx_loss.append(tf.reduce_mean(tf.multiply(tf.square(gradx1_s),tf.square(gradx2_s)),reduction_indices=[1,2,3])**0.25)
        grady_loss.append(tf.reduce_mean(tf.multiply(tf.square(grady1_s),tf.square(grady2_s)),reduction_indices=[1,2,3])**0.25)

        img1=tf.nn.avg_pool(img1, [1,2,2,1], [1,2,2,1], padding='SAME')
        img2=tf.nn.avg_pool(img2, [1,2,2,1], [1,2,2,1], padding='SAME')
    return gradx_loss,grady_loss

def compute_gradient(img):
    gradx=img[:,1:,:,:]-img[:,:-1,:,:]
    grady=img[:,:,1:,:]-img[:,:,:-1,:]
    return gradx,grady


def compute_percep_ncc_loss(input, output, reuse=False):
    weight_in = 1/(tf.reduce_max(tf.abs(input))+1e-10) 
    weight_out = 1/(tf.reduce_max(tf.abs(output))+1e-10) 
 
    input = input * weight_in
    output= output* weight_out
    zero_mat= tf.zeros(tf.shape(output),tf.float32)
    output  = tf.where(tf.greater(output,0),output,zero_mat)
    input  = tf.where(tf.greater(input,0),input,zero_mat)

    output= tf.pow(output,1/2.2)
    input = tf.pow(input,1/2.2)
 
    losses = [] 
    for l in range(3):
        losses.append(compute_pncc_loss(input,output))
        input=tf.nn.avg_pool(input, [1,2,2,1], [1,2,2,1], padding='SAME')
        output=tf.nn.avg_pool(output, [1,2,2,1], [1,2,2,1], padding='SAME')
    return sum(losses)/len(losses) 


def compute_gradient_loss(img1,img2,level=1):
    gradx_loss=[]
    grady_loss=[]
    
    for l in range(level):
        gradx1, grady1=compute_gradient(img1)
        gradx2, grady2=compute_gradient(img2)

        gradx_loss.append(tf.reduce_mean(tf.abs(gradx1-gradx2)))
        grady_loss.append(tf.reduce_mean(tf.abs(grady1-grady2)))

        img1=tf.nn.avg_pool(img1, [1,2,2,1], [1,2,2,1], padding='SAME')
        img2=tf.nn.avg_pool(img2, [1,2,2,1], [1,2,2,1], padding='SAME')
    return gradx_loss,grady_loss

def compute_ncc_loss(a, b):
    vector_a = slim.flatten(a)[0]
    vector_b = slim.flatten(b)[0]
    mean_a, var_a = tf.nn.moments(vector_a,axes=0)
    mean_b, var_b = tf.nn.moments(vector_b,axes=0)
    new_a = tf.divide((vector_a-mean_a),tf.sqrt(var_a)+1e-7)
    new_b = tf.divide((vector_b-mean_b),tf.sqrt(var_b)+1e-7)
    return tf.abs(tf.reduce_mean(new_a*new_b))


def compute_pncc_loss(input, output, reuse=False):
    vgg_real=build_vgg19(output*255.0,reuse=reuse)
    vgg_fake=build_vgg19(input*255.0,reuse=True)
    p1=compute_ncc_loss(vgg_real['conv1_2'],vgg_fake['conv1_2'])/2.6
    p2=compute_ncc_loss(vgg_real['conv2_2'],vgg_fake['conv2_2'])/4.8
    p3=compute_ncc_loss(vgg_real['conv3_2'],vgg_fake['conv3_2'])/3.7
    p4=compute_ncc_loss(vgg_real['conv4_2'],vgg_fake['conv4_2'])/5.6
    p5=compute_ncc_loss(vgg_real['conv5_2'],vgg_fake['conv5_2'])*10/1.5
    return  p2 +p3 +p4
