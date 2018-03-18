"""
Created on Fri Feb 16 03:59:12 2018

@author: HOSSAM ABDELHAMID
"""
"""This file includes CNN model for segmenting Retinal images"""
import tensorflow as tf

# Parameter: Num of Classes for segmentation
CLASS_NUM=2


# CNN Model 
# Data-set-> Input:x is grayscale (size=25*25), Output: ytruth (ground truth output)
# Prediction output: y 
class CNN(object):
    def __init__(self,batch_norm=True, whitening=False, is_training=True):
        self.x = tf.placeholder(tf.float32, shape=[None, 25, 25,1], name='x')
        self.ytruth = tf.placeholder(tf.float32, shape=[None, CLASS_NUM])
        ##########################First CNN Layer########################
        self.h_conv1 = tf.layers.conv2d(inputs=self.x,filters=12,kernel_size=[4, 4],padding="valid",activation=tf.nn.relu)
        self.pool1 = tf.layers.max_pooling2d(inputs=self.h_conv1, pool_size=[2, 2], strides=2)   
        ##########################Second CNN Layer########################
        self.h_conv2 = tf.layers.conv2d(inputs=self.pool1,filters=12,kernel_size=[4, 4],padding="valid",activation=tf.nn.relu)
        self.pool2 = tf.layers.max_pooling2d(inputs=self.h_conv2, pool_size=[2, 2], strides=2)
        ##########################Fully Connected Layer###################
        self.h_conv3 = tf.layers.conv2d(inputs=self.pool2,filters=100,kernel_size=[4, 4],padding="valid",activation=tf.nn.relu)
        self.h_conv3_flat = tf.reshape(self.h_conv3, [-1, 100])
        self.dense = tf.layers.dense(inputs=self.h_conv3_flat, units=CLASS_NUM, activation=tf.nn.relu)
        ##################################Output##########################
        self.y = self.dense
        
