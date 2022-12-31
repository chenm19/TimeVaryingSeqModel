import os
import tensorflow as tf
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random

#!/usr/bin/env python
# coding: utf-8

#!pip install scikit-fdaimport numpy as np
# !pip install scikit-fdaimport numpy as np
import skfda
from matplotlib import pyplot as plt
import numpy as np
# import torch
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import scipy.linalg
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

from tensorflow.keras import backend as K
from tensorflow.keras import activations
from tensorflow.keras import initializers
from tensorflow.keras.models import Sequential
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import constraints
tf.compat.v1.enable_eager_execution()
#np.random.seed(1234)


# Generated data


def full_data(n,t,basis,p):
    if basis == "fourier" :
        bas = skfda.representation.basis.Fourier(
        n_basis=p,
        domain_range=(min(t), max(t)))
    if basis == "bspline":
        bas = skfda.representation.basis.BSpline(
        n_basis=p,
        domain_range=(min(t), max(t)))
    if basis == "monomimal":
        bas = skfda.representation.basis.monomial(
        n_basis=p,
        domain_range=(min(t), max(t)))

    coef = np.random.normal(0, 0.5, size = (n,p))
    fd_basis = skfda.FDataBasis(basis=bas, coefficients= coef)

    return [fd_basis.evaluate(t), fd_basis]


### generating basis for weight update
## ln is the first time point, lb is the last one. We can hard code it to be 0 and 1 respectively since our time points are in [0,1]
## q is the number of basis functions
# t2 is the current time point
# t1 is the previous time point



def BasisExpansion(q, t, lb=0, ub=1):
    bas = skfda.representation.basis.BSpline(n_basis=q, domain_range=(lb, ub))
    #bas = skfda.representation.basis.Fourier(n_basis=q, domain_range=(lb, ub))
    Bt = np.squeeze(bas.evaluate(t))
    return Bt



# hyperparameter




n_train = 100  #     train  sample size
n_test = 100  #     test  sample size
p = 5  # number of basis functions
T = 40  # number of time points
t = np.linspace(0, 1, T)   # tim grid
basis = "fourier"   # choice of first basis
#basis2 = "bspline"  # choice of second basis
h = 32    # hidden layer numbers
activation = 'tanh'  #Activation function selection
lr = 1e-3   #learning rate
epoch = 30
num_layers = 1  #the number of layers
'''
To change the number of hidden layers, change the dimension of B. 
For example, if the number of hidden layers is, 
then B= scipy.linalg.block_diag(Bt, Bt, Bt, Bt, Bt)  
'''


epoch = epoch
batch_size = 32
steps_per_epoch = n_train // batch_size
metric = keras.metrics.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)






# Train date
data_full_train, data_fd_train = full_data(n = n_train, t = t, basis = basis, p = p) ## complete funcitonal data,  data_fd is the same as data_full  - it is just easier to plot
# Test  date
data_full_test, data_fd_test = full_data(n = n_test, t = t, basis = basis, p = p) ## complete funcitonal data,  data_fd is the same as data_full  - it is just easier to plot






x_full_train = np.delete(data_full_train,[T-1],1)
y_full_train = np.delete(data_full_train,[0],1)
#print(y_full_train)
#y_full_train = y_full_train[:,-1]
x_full_test = np.delete(data_full_test,[T-1],1)
true_full_test = np.delete(data_full_test,[0],1)




# Prepare train and test

print(x_full_train.shape)
print(y_full_train.shape)
#print(x_full_train)
#print(y_full_train)
print(x_full_test.shape)
print(true_full_test.shape)
# First, let's define a RNN Cell, as a layer subclass.


def plot(x, y, pred, index):  # index:int
    T_ = T
    t = np.linspace(0, 1, T_)
    fig, ax = plt.subplots()  # canvas size
    ax.plot(t[0:T_ - 1], y[index], '-', label='True')
    # ax.plot(t[0:T_-1],pred[index],label='Predict')
    # ax.plot(t[0:T_-1],x[index],'-*',label='Features')
    ax.legend()
    ax.set_title('Sample {}'.format(index))  # title
    # ax.set_xlabel('Time step') #x lable
    plt.show()



plot(y_full_train,y_full_train,y_full_train,0)


class MYLSTM_1(keras.layers.Layer):

    def __init__(self, units, q,
                 kernel_initializer='orthogonal',
                 **kwargs):
        super(MYLSTM_1, self).__init__()
        self.units = units
        self.state_size = units
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.q = q
        self.k = np.linspace(0, 1, T)
        self.m = 0
        self.t = 0
        self.update = 0
        self.steps = T

    def build(self, input_shape):
        super(MYLSTM_1, self).build(input_shape)
        self.Q1 = self.add_weight(shape=(input_shape[-1] + self.units, self.q * self.units),

                                  initializer=self.kernel_initializer,
                                  name='Wf',
                                  trainable=True)

        self.Q2 = self.add_weight(shape=(input_shape[-1] + self.units, self.q * self.units),

                                  initializer=self.kernel_initializer,
                                  name='Wi',
                                  trainable=True)

        self.Q3 = self.add_weight(shape=(input_shape[-1] + self.units, self.q * self.units),

                                  initializer=self.kernel_initializer,
                                  name='Wc',
                                  trainable=True)

        self.Q4 = self.add_weight(shape=(input_shape[-1] + self.units, self.q * self.units),

                                  initializer=self.kernel_initializer,
                                  name='Wo',
                                  trainable=True)

        # self.U = self.add_weight(
        # shape=(self.units, self.units),
        # initializer='uniform',
        # name='U')
        '''
        self.Q = self.add_weight(
            shape=(self.units, self.q * self.units),  
            initializer=self.kernel_initializer,
            name='theta')

        self.Q2 = self.add_weight(
            shape=(self.units, self.q * self.units), 
            initializer=self.kernel_initializer,
            name='theta2')
        '''
        self.built = True

    def call(self, inputs, states):
        # K.clear_session()
        xt = inputs
        Ht = states[0]
        Ct = states[0]

        self.t1 = self.k[self.m]
        Bt = BasisExpansion(self.q, self.t1)
        B = Bt
        for i in range(self.units - 1):
            B = scipy.linalg.block_diag(B, Bt)

        self.update = tf.matmul(self.Q1, B.T)
        self.BM = self.update  # Change matrix 1
        self.BM2 = tf.matmul(self.Q2, B.T)  # Change matrix 2
        self.BM3 = tf.matmul(self.Q3, B.T)
        self.BM4 = tf.matmul(self.Q4, B.T)

        cat = tf.concat((Ht, xt), -1)  # Concatenate Ht with input X
        Ft = tf.sigmoid(tf.matmul(cat, self.BM))  # Forget gate
        It = tf.sigmoid(tf.matmul(cat, self.BM2))  # Input gate
        Gt = tf.sigmoid(tf.matmul(cat, self.BM3))
        Ct = It * Gt + Ft * Ct  # Update the Ct
        Ot = tf.tanh(tf.matmul(cat, self.BM4))  # Output gate
        Ht = Ot * tf.tanh(Ct)  # Update the Ht
        Ct = tf.tanh(Ct)

        '''
        cat = tf.concat((Ht, inputs), -1)         # Concatenate Ht with input X
        Ft = tf.sigmoid(tf.matmul(cat, self.BM))  # Forget gate
        It = tf.sigmoid(tf.matmul(cat, self.BM2))  # Input gate
        Gt = tf.sigmoid(tf.matmul(cat, self.BM3))
        Ct = It * Gt + Ft * Ct  # update the Ct
        Ot = tf.tanh(tf.matmul(cat, self.BM4))     # Output  gate
        # Ht = Ot * tf.tanh(Ct)
        # Ct = tf.matmul(Ct, self.U)
        St = Ot * tf.tanh(Ct)
        Ht = tf.tanh(tf.matmul(St, self.BM))       # update the Ht
        Ct = tf.tanh(tf.matmul(Ct, self.BM2))      # update the Ct
        # Ct = tf.tanh(Ct)
        # print(Ct)
        '''
        self.m += 1
        if self.m == self.steps:
            self.m = 0
            self.update = 0

        return Ht, [Ct]


epoch_list5 = []
loss_list5 = []
model6 = Sequential()

for i in range(num_layers - 1):
    model6.add(RNN(MYLSTM_1(h, q=p, activation=activation), return_sequences=True))
model6.add(RNN(MYLSTM_1(h, q=p, activation=activation), return_sequences=True))
model6.add(tf.keras.layers.Dropout(0.2))
model6.add(tf.keras.layers.Dense(1))
'''
model5.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr,decay = 0.0),
               loss = 'mean_squared_error',metrics=['mean_squared_error'])
history5 = model5.fit(x_full_train,y_full_train,batch_size=batch_size,validation_data=(x_full_test,true_full_test),epochs=epoch,verbose=1)
'''
num_epoch = 1000  # number of training times
optimizer = tf.keras.optimizers.Adam(learning_rate=8e-4, decay=0.01)  # optimizer
for e in range(num_epoch):
    # Use tf.GradientTape() to record the gradient information of the loss function
    with tf.GradientTape() as tape:
        y_pred = model6(x_full_train)  # training set prediction
        loss = tf.reduce_mean(tf.square(y_pred - y_full_train))  # training set loss
    # TensorFlow automatically calculates the gradient of the loss function with respect to the independent variables (model parameters)
    grads = tape.gradient(loss, model6.trainable_variables)  # calculate gradients

    # TensorFlow automatically updates parameters based on gradient
    optimizer.apply_gradients(grads_and_vars=zip(grads, model6.trainable_variables))  # update parameters

    test_pred = model6(x_full_test)  # testing set prediction
    testloss = tf.reduce_mean(tf.square(test_pred - true_full_test))  # testing set loss
    if e % 100 == 0:
        epoch_list5.append(e)
        loss_list5.append(testloss.numpy())
        print('epochs : {}, train_loss : {}, test_loss : {}'.format(e, loss.numpy(), testloss.numpy()))









