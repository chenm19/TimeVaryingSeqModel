#Import libraries

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
#import os
import tensorflow as tf
from pathlib import Path
from tensorflow.keras import backend as K



import  numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle





#Import data set
#train=pd.read_csv('F:\\谷歌\\UCRArchive_2018\\PowerCons\\PowerCons_TRAIN.tsv', sep='\t', header=None)
df1=pd.read_excel('sin_cos.xlsx')
train = np.array(df1)
Train = np.expand_dims(train,axis=2)
train2 = np.delete(Train,0,axis=1)
train2 = np.delete(train2,-1,axis=1)
x_full_train= train2
print(train2.shape)
print(Train.shape)
#print(Train)
print(x_full_train.shape)
Y = Train[:,0]
print(Y.shape)
#print(Y)
y_full_train =  Y



#parameter

seed = 7
np.random.seed(seed)

p = 5  # number of basis functions

h =5  #hidden layer numbers
activation ='tanh'   #Activation function selection
lr = 1e-3   #    #learning rate
epoch = 50
batch_size = 64
num_layers = 1  #the number of layers
T = 49
t = np.linspace(0, 1, T)   # tim grid
def BasisExpansion(q, t, lb=0, ub=1):
    #bas = skfda.representation.basis.BSpline(n_basis=q, domain_range=(lb, ub))
    bas = skfda.representation.basis.Fourier(n_basis=q, domain_range=(lb, ub))
    Bt = np.squeeze(bas.evaluate(t))
    return Bt


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
        self.k = np.linspace(0, 1, T)
        self.m = 0
        self.update = 0
        self.steps = T

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1] + self.units, self.units),
                                 initializer=self.kernel_initializer,
                                 name='W')

        self.Q = self.add_weight(
            shape=(self.units, self.q * self.units),  # theta-matrix to be optimized
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
        self.BM = self.update  # BM is the external matrix and varies with the time step

        cat = tf.concat((Ht, inputs), -1)  # Concatenate Ht with input X
        St = tf.matmul(cat, self.W)
        Ht = tf.matmul(St, self.BM)
        if self.use_bias:
            Ht = Ht + self.bias
        if self.activation is not None:
            Ht = self.activation(Ht)
            # St = self.activation(St)
        self.m += 1
        if self.m == self.steps:
            self.m = 0
            self.update = 0

        return Ht, Ht


model3 = Sequential()

for i in range(num_layers-1):
    model3.add(RNN(MinimalRNNCell(h,q=p, activation=activation), return_sequences=True))
model3.add(RNN(MinimalRNNCell(h,q=p,activation=activation),return_sequences=False))
model3.add(tf.keras.layers.Dropout(0.2))
model3.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model3.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr,decay = 0.0)
                                            ,loss = 'binary_crossentropy',metrics=['accuracy'])

history3 = model3.fit(x_full_train,y_full_train,batch_size=batch_size,epochs=epoch,verbose=0)

