
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import skfda
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import activations
from tensorflow.keras import initializers
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import scipy.linalg
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import  StratifiedKFold
from sklearn.model_selection import KFold
#import os
import tensorflow as tf
from pathlib import Path
from tensorflow.keras import backend as K



import  numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle



# Display data
from sklearn import preprocessing

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('AGG')

import matplotlib.pyplot as plt

import tensorflow
import time



f=open('twitter.csv','rb')
df=pd.read_csv(f,encoding='gb18030')


data=np.array(df['High'])
print(data.shape)
data=data[::-1]

plt.figure()
plt.plot(data)
plt.show()

min_max_scaler = preprocessing.MinMaxScaler()
normalize_data = min_max_scaler.fit_transform(data.reshape(-1,1))


# Form training set

time_step=7

train_x,train_y=[],[]
for i in range(len(normalize_data)-time_step-1):
    x=normalize_data[i:i+time_step]
    y=normalize_data[i+1:i+time_step+1]
    train_x.append(x.tolist())
    train_y.append(y.tolist())

x_full_train = np.array(train_x)
y_full_train = np.array(train_y)
#y_full_train = np.squeeze(y_full_train,axis=2)
print(x_full_train.shape)
print(y_full_train.shape)


# Hyperparameter


def BasisExpansion(q, t, lb=0, ub=1):
    bas = skfda.representation.basis.BSpline(n_basis=q, domain_range=(lb, ub))
    #bas = skfda.representation.basis.Fourier(n_basis=q, domain_range=(lb, ub))
    Bt = np.squeeze(bas.evaluate(t))
    return Bt





p = 5  # number of basis functions
T = 7  # number of time points
t = np.linspace(0, 1, T)   # tim grid
basis = "bspline"   # choice of first basis
#basis2 = "bspline"  # choice of second basis
h = 10    # hidden layer numbers
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
metric = keras.metrics.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)



class MinimalRNNCell(keras.layers.Layer):

    def __init__(self, units, q,
                 activation=None,
                 kernel_initializer='orthogonal',
                 use_bias=True,
                 **kwargs):
        self.units = units
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.q = q
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.state_size = units
        super(MinimalRNNCell, self).__init__(**kwargs)
        self.k =  np.linspace(0, 1, T)
        self.m = 0
        self.update = 0
        self.steps = T

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1] + self.units, self.units),
                                 initializer=self.kernel_initializer,
                                 name='W')

        self.Q = self.add_weight(
            shape=(self.units, self.q * self.units ),  # theta-matrix to be optimized
            initializer=self.kernel_initializer,
            name='theta')
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer='uniform',
                                        name='bias')

        self.built = True

    def call(self, inputs, states):
        # K.clear_session()
        Ht = states[0]
        # h = tf.matmul(inputs, self.W)
        self.t1 = self.k[self.m]
        Bt = BasisExpansion(self.q, self.t1)
        B = Bt
        for i in range(self.units - 1):
            B = scipy.linalg.block_diag(B, Bt)



        self.update = tf.matmul(self.Q, B.T)
        self.BM = self.update          # The Change matrix
        cat = tf.concat((Ht, inputs), -1)
        Ht = tf.matmul(cat, self.W)
        Ht = tf.matmul(Ht, self.BM)    # Update the Ht
        if self.use_bias:
            Ht = Ht + self.bias
        if self.activation is not None:
            Ht = self.activation(Ht)
            #St = self.activation(St)
        self.m += 1
        if self.m == self.steps:
            self.m = 0
            self.update = 0

        return Ht, [Ht]


epoch_list3 = []
loss_list3 = []
model3 = Sequential()

for i in range(num_layers - 1):
    model3.add(RNN(MinimalRNNCell(h, q=p, activation=activation), return_sequences=True))
model3.add(RNN(MinimalRNNCell(h, q=p, activation=activation), return_sequences=True))
model3.add(tf.keras.layers.Dropout(0.2))
model3.add(tf.keras.layers.Dense(1))
'''
model3.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr,decay = 0.0),
               loss = 'mean_squared_error',metrics=['mean_squared_error'])
history3 = model3.fit(x_full_train,y_full_train,batch_size=batch_size,validation_data=(x_full_test,true_full_test),epochs=epoch,verbose=1)
'''
num_epoch = 1000  # number of training times
optimizer = tf.keras.optimizers.Adam(learning_rate=8e-4)  # optimizer
for e in range(num_epoch):
    # Use tf.GradientTape() to record the gradient information of the loss function
    with tf.GradientTape() as tape:
        y_pred = model3(x_full_train)  # training set prediction
        loss = tf.reduce_mean(tf.square(y_pred - y_full_train))  # training set loss
    # TensorFlow automatically calculates the gradient of the loss function with respect to the independent variables (model parameters)
    grads = tape.gradient(loss, model3.trainable_variables)  # calculate gradients

    # TensorFlow automatically updates parameters based on gradient
    optimizer.apply_gradients(grads_and_vars=zip(grads, model3.trainable_variables))  # update parameters

    test_pred = model3(x_full_train)  # testing set prediction
    testloss = tf.reduce_mean(tf.square(test_pred - y_full_train))  # testing set loss
    if e % 100 == 0:
        epoch_list3.append(e)
        loss_list3.append(testloss.numpy())
        print('epochs : {}, train_loss : {}, test_loss : {}'.format(e, loss.numpy(), testloss.numpy()))

















