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





class MyRNNCell(keras.layers.Layer):

    def __init__(self, units,
                 activation = None,
                 kernel_initializer='orthogonal',
                 use_bias=True,
                 **kwargs):
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.state_size = units


        super(MyRNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1] + self.units, self.units),
                                 initializer=self.kernel_initializer,
                                 name='W')
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer='uniform',
                                        name='bias')

        self.built = True

    def call(self, inputs, states):
        Ht = states[0]
        cat = tf.concat((Ht, inputs), -1)  # Concatenate Ht with input X
        Ht = tf.matmul(cat, self.W)
        #print(Ht.shape)
        if self.use_bias:
            Ht = Ht + self.bias
        if self.activation is not None:
            Ht = self.activation(Ht)

        return Ht, [Ht]


epoch_list1 = []
loss_list1 = []
model8 = Sequential()

for i in range(num_layers - 1):
    model8.add(RNN(MyRNNCell(h, activation=activation), return_sequences=True))
model8.add(RNN(MyRNNCell(h, activation=activation), return_sequences=True))
model8.add(tf.keras.layers.Dropout(0.2))
model8.add(tf.keras.layers.Dense(1))
'''
model8.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr,decay = 0.0),
               loss = 'mean_squared_error',metrics=['mean_squared_error'])
history8 = model8.fit(x_full_train,y_full_train,batch_size=batch_size,validation_data=(x_full_test,true_full_test),epochs=epoch,verbose=1)
'''
num_epoch = 1000  # number of training times
optimizer = tf.keras.optimizers.Adam(learning_rate=8e-4, decay=0.01)  # optimizer
epoch_list1 = []
loss_list1 = []
for e in range(num_epoch):
    # Use tf.GradientTape() to record the gradient information of the loss function
    with tf.GradientTape() as tape:
        y_pred = model8(x_full_train)  # training set prediction
        loss = tf.reduce_mean(tf.square(y_pred - y_full_train))  # training set loss
    # TensorFlow automatically calculates the gradient of the loss function with respect to the independent variables (model parameters)
    grads = tape.gradient(loss, model8.trainable_variables)  # calculate gradients
    # print('before:',grads[0].numpy(),end='\t')

    # grads=[tf.clip_by_value(g,clip_value_min=-1e3, clip_value_max=6e-6, name=None)
    # for g in grads] #gradient clipping
    # print('after:',grads[0].numpy(),end='\t')
    # print('loss:',loss)

    # TensorFlow automatically updates parameters based on gradient
    optimizer.apply_gradients(grads_and_vars=zip(grads, model8.trainable_variables))  # update parameters

    test_pred = model8(x_full_test)  # testing set prediction
    testloss = tf.reduce_mean(tf.square(test_pred - true_full_test))  # testing set loss
    if e % 100 == 0:
        epoch_list1.append(e)
        loss_list1.append(testloss.numpy())
        print('epochs : {}, train_loss : {}, test_loss : {}'.format(e, loss.numpy(), testloss.numpy()))

















