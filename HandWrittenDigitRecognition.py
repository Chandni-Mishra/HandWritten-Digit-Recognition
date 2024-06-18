# Code without using library functions


import matplotlib 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2 as cv
import tensorflow as tf


mnist = tf.keras.datasets.mnist
(x_train,y_train) , (x_test,y_test) = mnist.load_data()

class nnetwork:
    def __init__(self):
        self.w1 = np.random.randn(20,784) * np.sqrt(1. / 20)
        self.w2 = np.random.randn(10,20) * np.sqrt(1. / 10)
        self.learning_rate = 0.001
        self.epocs = 10
    
    def sigmoid(self,x):
        return (1 / 1 + np.exp(-x))
    
    def diff_sigmoid(self,x):
        return (self.sigmoid(x)*(1 - self.sigmoid(x)))
    
    def softmax(self,x):
        exps = np.exp(x - x.max())
        return exps / np.sum(exps,axis = 0)
    
    def diff_softmax(self,x):
        exps = np.exp(x - x.max())
        return exps / np.sum(exps,axis = 0) * (1 - exps / np.sum(exps,axis = 0))
        
    def forward_propagation(self,x):
        self.h_in = np.dot(self.w1,x)
        self.h_out = self.sigmoid(self.h_in)
        
        self.o_in = np.dot(self.w2,self.h_out)
        self.out = self.softmax(self.o_in)
        
        return self.out
    
    def backward_propagation(self,output,y,x):
        error1 = 2 * (output - y) / output.shape[0] * self.diff_softmax(self.o_in)
        change_w2 = np.outer(error1,self.h_out)

        error2 = np.dot(self.w2.T,error1) * self.diff_sigmoid(self.h_in)
        change_w1 = np.outer(error2,x)
        self.w2 = self.w2 - self.learning_rate * change_w2
        self.w1 = self.w1 - self.learning_rate * change_w1

    def cal_accuracy(self,x_test,y_test):
        accuracy = []
        for i in range(10000):
            x = x_test[i].flatten()
            x = (((x/ 255.0) * 0.99) + 0.01)
            output = self.forward_propagation(x)
            y = np.zeros(10) + 0.01
            y[int(y_test[i])] = 0.99
            comp = (np.argmax(output) == np.argmax(y))
            accuracy.append(comp)
        return accuracy
    
    def train_network(self,x_train,y_train):
        for epoc in range(self.epocs):
            for i in range(60000):
                y = np.zeros(10) + 0.01
                y[int(y_train[i])] = 0.99
                x = x_train[i].flatten()
                x = (((x/ 255.0) * 0.99) + 0.01)
                output = self.forward_propagation(x)
                self.backward_propagation(output,y,x)
            
            accuracy = self.cal_accuracy(x_test,y_test)
            print(f'epoc = {epoc+1} , Accuracy = {np.mean(accuracy) * 100}%')
    
Ann = nnetwork()
Ann.train_network(x_train,y_train) 

x = x_test[544]
x = x.flatten()
x = x / 255.0 * 0.99 + 0.01
out = Ann.forward_propagation(x)
print(np.argmax(out))
