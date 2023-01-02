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




class MYLSTM1(keras.layers.Layer):

    def __init__(self, units,
                 kernel_initializer='orthogonal',
                 **kwargs):
        super(MYLSTM1, self).__init__()
        self.units = units
        self.state_size = units
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        super(MYLSTM1, self).build(input_shape)
        self.Wf = self.add_weight(shape=(input_shape[-1] + self.units, self.units),

                                  initializer=self.kernel_initializer,
                                  name='Wf',
                                  trainable=True)

        self.Wi = self.add_weight(shape=(input_shape[-1] + self.units, self.units),

                                  initializer=self.kernel_initializer,
                                  name='Wi',
                                  trainable=True)

        self.Wc = self.add_weight(shape=(input_shape[-1] + self.units, self.units),

                                  initializer=self.kernel_initializer,
                                  name='Wa',
                                  trainable=True)

        self.Wo = self.add_weight(shape=(input_shape[-1] + self.units, self.units),

                                  initializer=self.kernel_initializer,
                                  name='Wo',
                                  trainable=True)

        self.built = True

    def call(self, inputs, states):
        xt = inputs
        Ht = states[0]
        Ct = states[0]
        cat = tf.concat((Ht, xt), -1)             # Concatenate Ht with input X
        Ft = tf.sigmoid(tf.matmul(cat, self.Wf))  # Forget gate
        It = tf.sigmoid(tf.matmul(cat, self.Wi))  # Input gate
        Gt = tf.sigmoid(tf.matmul(cat, self.Wc))
        Ct = It * Gt + Ft * Ct                    # Update the Ct
        Ot = tf.tanh(tf.matmul(cat, self.Wo))     # Output gate
        Ht = Ot * tf.tanh(Ct)                     # Update the Ht
        Ct = tf.tanh(Ct)
        return Ht, Ct



model5 = Sequential()
for i in range(num_layers-1):
    model5.add(RNN(MYLSTM1(h,q=p), return_sequences=True))
model5.add(RNN(MYLSTM1(h,q=p),return_sequences=False))
model5.add(tf.keras.layers.Dropout(0.2))
model5.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model5.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr,decay = 0.0),
               loss = 'binary_crossentropy',metrics=['accuracy'])
history5 = model5.fit(x_full_train,y_full_train,batch_size=batch_size,epochs=epoch,verbose=0)