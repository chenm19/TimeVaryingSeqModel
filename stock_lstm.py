
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
        return Ht, [Ct]


epoch_list4 = []
loss_list4 = []
model5 = Sequential()

for i in range(num_layers - 1):
    model5.add(RNN(MYLSTM1(h, activation=activation), return_sequences=True))
model5.add(RNN(MYLSTM1(h, activation=activation), return_sequences=True))
model5.add(tf.keras.layers.Dropout(0.2))
model5.add(tf.keras.layers.Dense(1))
'''
model5.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr,decay = 0.0),
               loss = 'mean_squared_error',metrics=['mean_squared_error'])
history5 = model5.fit(x_full_train,y_full_train,batch_size=batch_size,validation_data=(x_full_test,true_full_test),epochs=epoch,verbose=1)
'''
num_epoch = 1000  # number of training times
optimizer = tf.keras.optimizers.Adam(learning_rate=8e-4)  # optimizer
for e in range(num_epoch):
    # Use tf.GradientTape() to record the gradient information of the loss function
    with tf.GradientTape() as tape:
        y_pred = model5(x_full_train)  # training set prediction
        loss = tf.reduce_mean(tf.square(y_pred - y_full_train))  # training set loss
    # TensorFlow automatically calculates the gradient of the loss function with respect to the independent variables (model parameters)
    grads = tape.gradient(loss, model5.trainable_variables)  # calculate gradients

    # TensorFlow automatically updates parameters based on gradient
    optimizer.apply_gradients(grads_and_vars=zip(grads, model5.trainable_variables))  # update parameters

    test_pred = model5(x_full_train)  # testing set prediction
    testloss = tf.reduce_mean(tf.square(test_pred - y_full_train))  # testing set loss
    if e % 100 == 0:
        epoch_list4.append(e)
        loss_list4.append(testloss.numpy())
        print('epochs : {}, train_loss : {}, test_loss : {}'.format(e, loss.numpy(), testloss.numpy()))





















