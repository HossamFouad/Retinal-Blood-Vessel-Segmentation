"""
Created on Fri Feb 16 03:59:12 2018

@author: HOSSAM ABDELHAMID
"""

"""This file includes the preprocessing for images"""
import scipy.misc
import random
import csv
DATA_DIR = '../Database/'
import tensorflow as tf
import numpy as np

# One Hot Matrix
# hot encoding: 0-> [1 0] and 1-> [0 1]
def one_hot_matrix(labels, C):
    """
    Creates a matrix where the i-th row corresponds to the ith class number and the jth column
                     corresponds to the jth training example. So if example j had a label i. Then entry (i,j) 
                     will be 1. 
                     
    Arguments:
    labels -- vector containing the labels 
    C -- number of classes, the depth of the one hot dimension
    
    Returns: 
    one_hot -- one hot matrix
    """
      
    # Create a tf.constant equal to C (depth), name it 'C'. (approx. 1 line)
    C = tf.constant(C, name = "C")
    
    # Use tf.one_hot, be careful with the axis (approx. 1 line)
    one_hot_matrix = tf.one_hot(labels,C,axis=1)
    
    # Create the session (approx. 1 line)
    sess = tf.Session()
    
    # Run the session (approx. 1 line)
    one_hot = sess.run(one_hot_matrix)
    
    # Close the session (approx. 1 line). See method 1 above.
    sess.close()
       
    return one_hot



# Data Handling, shuffling and loading
class DataReader(object):
    def __init__(self, data_dir=DATA_DIR,sequential=False):
        self.load()

    def load(self):
        ########################Train Data#########################
        xs = []     #input data
        ys = []     #output Label 
        yss=[]
        # batch_pointer variables are used to iterate cyclic on the data batches
        self.train_batch_pointer = 0 
        self.test_batch_pointer = 0
        self.total = 0  # count number of training samples
        # CVS file that has all images names and classes of training data
        with open('TRAIN.csv') as f:
            reader = csv.DictReader(f)
            print("Fetching Data...")
            for row in reader:
                xs.append(DATA_DIR+row['image']+'.tif')
                yss.append((row['classes']))
                self.total += 1

       
        print('Total training data: ' + str(self.total))
        ys=one_hot_matrix(np.float32(yss),2)
        self.num_images = len(xs)
        self.train_xs=xs
        self.train_ys=yss
        c = list(zip(xs, ys))
        random.shuffle(c)
        # Random Data xs->images , ys->ouptut labels one hot encoded matrix 
        self.train_xs, self.train_ys = zip(*c)
        self.train_ys=np.int32(self.train_ys)
        
        ########################Test Data#########################
        xtest = []     #input data
        ytest = []     #output Label 
        ysstest=[]
        self.total_test = 0 # count num of test data
        # CVS file that has all images names and Class of testing data
        with open('TEST.csv') as f:
            reader = csv.DictReader(f)
            print("Fetching Testing Data ...")
            for row in reader:
                xtest.append(DATA_DIR+row['image']+'.tif')
                ysstest.append((row['classes']))
                self.total_test += 1
                
                
        print('Total test data: ' + str(self.total_test))
        ytest=one_hot_matrix(np.float32(ysstest),2)
        c1 = list(zip(xtest, ytest))
        # Random Data xs->images , ys->ouptut labels one hot encoded matrix 
        random.shuffle(c1)
        self.test_xs, self.test_ys = zip(*c1)
        self.test_ys=np.int32(self.test_ys)
        self.test_xs=xtest
        self.test_ys=ysstest
        
        # Get Random mini batch of size batch_size
    def load_train_batch(self, batch_size):
        x_out = np.zeros((batch_size,25,25,1))
        y_out = np.zeros((batch_size,2))
        image = np.zeros((25,25,1))
        for i in range(0, batch_size):
            image = scipy.misc.imread(self.train_xs[(self.train_batch_pointer + i) % self.num_images])
            image=image.reshape(25,25,1)
            x_out[i,:,:,:]=image / 255.0
            y_out[i,:]=self.train_ys[(self.train_batch_pointer + i) % self.num_images]
        self.train_batch_pointer += batch_size
        return x_out, y_out
    
    def load_test_data(self, test_size):
        x_out = np.zeros((test_size,25,25,1))
        y_out = np.zeros((test_size,2))
        image = np.zeros((25,25,1))
        for i in range(0, test_size):
            image = scipy.misc.imread(self.test_xs[(self.test_batch_pointer + i) % self.total_test])
            image=image.reshape(25,25,1)
            x_out[i,:,:,:]=image / 255.0
            y_out[i,:]=self.test_ys[(self.test_batch_pointer + i) % self.total_test]
        self.test_batch_pointer += test_size
        return x_out, y_out

