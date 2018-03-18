
# coding: utf-8

# In[11]:

import os
import tensorflow as tf
from Model import CNN 
from preprocessing import DataReader  
import scipy.misc
import numpy as np


'''Saving Model'''
LOGDIR = 'vol'
CHECKPOINT_EVERY = 10
RESTORE_FROM='model-step-770-val-0.397487.ckpt'
print_cost = True

def main():
    print("Testing...")
    #CNN_Model
    '''initialization'''
    tf.reset_default_graph()
    sess = tf.InteractiveSession()  
    # Forward propagation      
    model = CNN()

    #data reading
    data_reader = DataReader()

    #intialize saving
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    sess.run(init)
  #restoring the model
    if RESTORE_FROM is not None:
        saver.restore(sess, os.getcwd()+'\\'+LOGDIR+'\\'+RESTORE_FROM)
        print('Model restored from ' + os.getcwd()+'\\'+LOGDIR+'\\'+RESTORE_FROM)
    count=0   
    example_num=50799  # num of exapmles
    feature1=np.zeros((example_num,1452)) 
    feature2=np.zeros((example_num,192))
    feature3=np.zeros((example_num,100))
    label=np.zeros((example_num,1))
    image = np.zeros((1,25,25,1))
    for i in range(0, example_num):
            image = scipy.misc.imread(data_reader.test_xs[i])
            image=image.reshape(1,25,25,1)/ 255.0
            output1 = model.pool1.eval(feed_dict={model.x: image})
            feature1[i,:]= np.reshape(output1, [-1, 1452])
            output2 = model.pool2.eval(feed_dict={model.x: image})
            feature2[i,:]= np.reshape(output2, [-1, 192])
            output3 = model.h_conv3_flat.eval(feed_dict={model.x: image})
            feature3[i,:]= output3
            label[i]=data_reader.test_ys[i]
            count=count+1
            print(count)
    scipy.io.savemat(os.getcwd()+'\\'+'feature1_test.mat', mdict={'arr': feature1})
    scipy.io.savemat(os.getcwd()+'\\'+'feature2_test.mat', mdict={'arr': feature2})
    scipy.io.savemat(os.getcwd()+'\\'+'feature3_test.mat', mdict={'arr': feature3})
    scipy.io.savemat(os.getcwd()+'\\'+'label_test.mat', mdict={'arr': label})
    
if __name__ == '__main__':
    main()