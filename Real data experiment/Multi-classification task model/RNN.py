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
df1=pd.read_excel('sin_cos_tan.xlsx')
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

#print(Y)
y_full_train =  Y
print(y_full_train.shape)


#parameter

seed = 7
np.random.seed(seed)

p = 8                        # number of basis functions

h =5                       #hidden layer numbers
activation = 'tanh'          #Activation function selection
lr = 1e-3   #               #learning rate
epoch = 60
num_layers = 1              #the number of layers
T = 49
t = np.linspace(0, 1, T)    # tim grid
batch_size= 128
num_classes=3               #number of classes
def BasisExpansion(q, t, lb=0, ub=1):
    #bas = skfda.representation.basis.BSpline(n_basis=q, domain_range=(lb, ub))
    bas = skfda.representation.basis.Fourier(n_basis=q, domain_range=(lb, ub))
    Bt = np.squeeze(bas.evaluate(t))
    return Bt




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

model2 = Sequential()

for i in range(num_layers - 1):
    model2.add(RNN(MyRNNCell(h, activation=activation), return_sequences=True))
model2.add(RNN(MyRNNCell(h, activation=activation), return_sequences=False))
model2.add(tf.keras.layers.Dropout(0.2))
model2.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

model2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr, decay=0.00),
                   loss='categorical_crossentropy', metrics=['accuracy'])
y_train =  to_categorical(y_full_train, num_classes=num_classes)
history2=model2.fit(x_full_train, y_train, epochs=epoch, batch_size=batch_size, verbose=0)
