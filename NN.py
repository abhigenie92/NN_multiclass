from __future__ import print_function
import pandas as pd
import numpy as np
import pdb,math
import matplotlib.pyplot as plt

class NN(object):
# initialization code
    def __init__(self,train_data,train_labels,hidden_layer_units=100,learning_rate=0.005):
        #  parameter initialization
        self.train_data=train_data
        self.train_labels=train_labels
        num_samples, num_features=self.train_data.shape
        num_classes=self.train_labels.ix[:,0].nunique()
        self.num_samples=num_samples
        self.learning_rate=learning_rate

        # number of neurons in each layer
        self.input_layer=num_features
        self.hidden_layer=hidden_layer_units
        self.output_layer=num_classes

        #initialise the weights
        self.w1 = 0.01 * np.random.randn(self.input_layer, self.hidden_layer)
        self.w2 = 0.01 * np.random.randn(self.hidden_layer, self.output_layer)

        #initialise the baises
        self.b1 = np.zeros((1,self.hidden_layer))
        self.b2 = np.zeros((1,self.output_layer))


# forward pass code
    def forward(self,x_input):
        '''feed-forward the input(passed as x_input)'''
        #1st layer
        self.z2 = np.dot(x_input, self.w1) +self.b1 # pre-activation for the 1st layer
        self.a2 = self.sigmoid(self.z2)             # activation function o/p 1st layer

        #2nd layer
        self.z3 = np.dot(self.a2, self.w2) +self.b2 # pre-activation for the 2nd layer
        self.a3 = self.softmax(self.z3)             # activation function o/p 2nd layer (probabilites)

        #return the output probabilites for each class
        return self.a3

    def softmax(self,x):
        '''computes the sigmoid activation function'''
        x=np.exp(x)                   # raise to exponent
        x = x / x.sum(axis=1)[:,None] # divide by summation over all classes
        return x

    def sigmoid(self,z):
        '''computes the sigmoid activation function'''
        return 1/(1+np.exp(-z))

# back-prop. code
    def derivative_sigmoid(self,z):
        '''computes the derivative of sigmoid activation function'''
        return np.exp(-z)/((1+np.exp(-z))**2)

    def back_prop(self):
        ''' returns the derivatives for weights and biases'''
        # 2nd layer
        delta_scores = self.forward(self.train_data) # the output of the neural network.
                                                     # A prob. for each class M X N.
        delta_scores[range(self.num_samples),self.train_labels-1] -= 1
        backprop_error_3 = delta_scores / self.num_samples
        dw2 = np.dot(self.a2.T, backprop_error_3)
        db2 = np.sum(backprop_error_3, axis=0, keepdims=True)

        #1st layer
        delta_hidden=self.derivative_sigmoid(self.z2)
        backprop_error_2 = np.dot(backprop_error_3, self.w2.T)*delta_hidden
        dw1 = np.dot(self.train_data.T, backprop_error_2)
        db1 = np.sum(delta_hidden, axis =0, keepdims=True)
        return dw1, dw2, db1, db2

    def update(self,dw1, dw2, db1, db2):
        '''updates the weights and biases'''
        self.w1 += -self.learning_rate * dw1
        self.w2 += -self.learning_rate * dw2
        self.b1 += -self.learning_rate * db1
        self.b2 += -self.learning_rate * db2

# accuracy checking and training code
    def train(self):
        ''' trains the neural network, calling backprop() and updating the weight with update()'''
        for i in range(1000):
            if (i % 1==0):
                self.cross_entropy_and_accuracy()
                print ("Iter: %d, loss & train_acc_per (%f,%f) " %(i, *self.cross_entropy_and_accuracy()))
            dw1, dw2, db1, db2 = self.back_prop()
            self.update(dw1, dw2, db1, db2)

    def cross_entropy_and_accuracy(self):
        '''
        1. computes the cross-entropy cost-function for train_data and train_labels
        2. computes the current accuracy/training error
        '''
        X=self.train_data
        y=self.train_labels
        num_samples=X.shape[0]

        # cross-entropy computation
        probs=self.forward(X)
        y=y-1           # y ranges in 1-26, subtract 1 for indexing to make range 0-25
        logs=-np.log(probs[np.arange(num_samples),y])
        cross_entropy=np.mean(logs)

        # training error computation
        predicted_labels=np.argmax(probs, axis=1)+1
        training_error=np.mean(y.ix[:,0]==predicted_labels)

        return cross_entropy, training_error

if __name__ == "__main__":
    train_data=pd.read_csv('train_data.csv')
    train_labels=pd.read_csv('train_labels.csv')
    NN_obj = NN(train_data,train_labels) #15000 x 16
    NN_obj.train()
