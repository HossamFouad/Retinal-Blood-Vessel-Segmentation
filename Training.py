"""
Created on Fri Feb 16 03:59:12 2018

@author: HOSSAM ABDELHAMID
"""

import os
import tensorflow as tf
import time
from Model import CNN 
from preprocessing import DataReader  
import numpy as np
import matplotlib.pyplot as plt


# HyperParameters
'''Steps NUM = epoch_num*batch_num'''
BATCH_SIZE = 256
TEST_SIZE= 50
'''Steps NUM = epoch_num*batch_num'''
EPOCH_NUM=10
#start iteration
START_STEP = 0
'''Learning Rate''' 
LEARNING_RATE = 1e-4
'''Saving Model'''
LOGDIR = 'vol'
CHECKPOINT_EVERY = 10
RESTORE_FROM='model-step-377-val-0.167728.ckpt'
print_cost = True
# In[3]:
# used Functions
def compute_cost(Z3, Y):
    """
    Computes the cost
    
    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3
    
    Returns:
    cost - Tensor of the cost function
    """
    
    cost = tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = Y)
    return cost

# Training Environment
# -------------------

# In[4]:


def main():
    print("Start training Model...")
    '''initialization'''
    tf.reset_default_graph()
    sess = tf.InteractiveSession()  
    # Forward propagation      
    model = CNN()
    # To keep track of the cost
    costs = []         

    # Cost function
    with tf.name_scope('loss'):
        loss = compute_cost(model.y,model.ytruth)
    loss = tf.reduce_mean(loss)
    
    #data reading
    data_reader = DataReader()
    
    #calculate number of iterations per epoch
    NUM_BATCHES_PER_EPOCH=int(data_reader.num_images/BATCH_SIZE)
    print('Num of batches per epoch :',NUM_BATCHES_PER_EPOCH)
    NUM_TEST_DATA=int(data_reader.total_test/TEST_SIZE)
    NUM_STEPS=NUM_BATCHES_PER_EPOCH*EPOCH_NUM
    print("Total No. of iterations :",NUM_STEPS)

    # Optimizer
    with tf.name_scope('adam'):
        train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
        
    #prediction
    correct_prediction = tf.equal(tf.argmax(model.y,1), tf.argmax(model.ytruth,1))
    with tf.name_scope('accuracy'):
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    #intialize saving
    saver = tf.train.Saver()
    

    min_loss = 1.0
    start=0
    init = tf.global_variables_initializer()
    sess.run(init)
    steps=0
    test_error=0
  #restoring the model
    if RESTORE_FROM is not None:
        saver.restore(sess, os.getcwd()+'\\'+LOGDIR+'\\'+RESTORE_FROM)
        print('Model restored from ' + os.getcwd()+'\\'+LOGDIR+'\\'+RESTORE_FROM)
       
    
    for epoch in range(EPOCH_NUM):
        minibatch_cost = 0.0
        for i in range(START_STEP, NUM_BATCHES_PER_EPOCH):
            start = time.time()
            steps+=1
            #get minibatch
            xs, ys = data_reader.load_train_batch(BATCH_SIZE)
            # run session
            _ , temp_cost = sess.run([train_step,loss], feed_dict={model.x: xs, model.ytruth: ys})
            #evauate train error
            train_error = loss.eval(feed_dict={model.x: xs, model.ytruth: ys})
            #evaluate average error per epoch
            minibatch_cost += temp_cost / NUM_BATCHES_PER_EPOCH
            #evaluate train accuracy
            train_accuracy = accuracy.eval({model.x: xs, model.ytruth: ys})
            end = time.time()
            elapsed = end - start
            
            print("Step%d [Train Loss= %g ,Accuracy= %g, elapse= %g min]"  % (steps,train_error,train_accuracy*100,elapsed*(NUM_STEPS-steps)/60))
            
            if steps% 100 == 0 or steps==NUM_BATCHES_PER_EPOCH*EPOCH_NUM-1:
                test_cost = 0.0
                test_acc=0.0
                for j in range(NUM_TEST_DATA):
                    xtest, ytest = data_reader.load_test_data(TEST_SIZE)
                    test_error = loss.eval(feed_dict={model.x: xtest, model.ytruth: ytest})
                    test_cost +=  test_error/ NUM_TEST_DATA
                    test_accuracy = accuracy.eval({model.x: xtest, model.ytruth: ytest})
                    test_acc +=  test_accuracy/ NUM_TEST_DATA
                print("Testing... Test Loss= %g  Accuracy:= %g" % (test_cost,test_acc*100))

            #saving 
            if steps > 0 and steps % CHECKPOINT_EVERY == 0:
                if not os.path.exists(LOGDIR):
                    os.makedirs(LOGDIR)
                checkpoint_path = os.path.join(LOGDIR, "model-step-%d-val-%g.ckpt" % (i, test_error))
                filename = saver.save(sess, checkpoint_path)
                print("Model saved in file: %s" % filename)
                if test_error < min_loss:
                    min_loss = test_error
                    if not os.path.exists(LOGDIR):
                        os.makedirs(LOGDIR)
                    checkpoint_path = os.path.join(LOGDIR, "model-step-%d-val-%g.ckpt" % (i, test_error))
                    filename = saver.save(sess, checkpoint_path)
                    print("Model saved in file: %s" % filename)

        if print_cost == True and epoch % 5 == 0:
            print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
        if print_cost == True and epoch % 1 == 0:
            costs.append(minibatch_cost)

        
    checkpoint_path = os.path.join(LOGDIR, "model-step-final.ckpt")
    filename = saver.save(sess, checkpoint_path)
    print("Model saved in file: %s" % filename)
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.show()  
    
if __name__ == '__main__':
    main()
