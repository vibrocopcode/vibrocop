#!/usr/bin/env python
# coding: utf-8

# In[156]:


import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
df = pd.read_csv('op7tloud.csv', header=None, names=['smean', 'smax', 'smin', 'std_dev', 'variance','range','cv','skewness','kurtosis','q25','q50','q75','mean_crossing_rate','power','entropy','frequency_ratio','irrk','irrj','sharpness','smoothness','specCentroid','specstddev','specCrest','specSkewness','specKurt','maxfreq','numPeaks','zeroCrossingRate','slopeChanges','numInflectionPoint','totalPeak','silentTime','d1','d2','d3','d4','d5','d6','d7','d8','d9','d10','class'])


# In[157]:


df.tail()


# In[ ]:





# In[158]:


df['labels'] =df['class'].astype('category').cat.codes


# In[159]:


df=df.drop(['sharpness'], axis='columns')


# In[160]:


df=df.drop(['q75'], axis='columns')


# In[161]:


df.head()


# Start preprocess

# Preprocessing Test

# In[162]:


#df['smean'] = df['smean']/df['smean'].max()
#df['smean'] = (df['smean'] - df['smean'].min())/(df['smean'].max() - df['smean'].min())
#df['smoothness'] = (df['smoothness'] - df['smoothness'].min())/(df['smoothness'].max() - df['smoothness'].min())
#df['smax'] = (df['smax']-df['smax'].mean())/df['smax'].std()


# In[163]:


df.head()


# In[164]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from tensorflow import keras


# In[165]:


#Some Preprocessing TEST
#clipping test


# In[166]:


X = df[['smean', 'smax', 'smin', 'std_dev', 'variance','range','cv','skewness','kurtosis','q25','q50','mean_crossing_rate','power','entropy','frequency_ratio','irrk','irrj','smoothness','specCentroid','specstddev','specCrest','specSkewness','specKurt','numPeaks','zeroCrossingRate','slopeChanges','numInflectionPoint','totalPeak','silentTime','d1','d2','d3','d4','d5','d6','d7','d8','d9','d10','maxfreq']]
Y = df['labels']
x_train, x_test, y_train, y_test = train_test_split(np.asarray(X), np.asarray(Y), test_size=0.2, shuffle= True)


# In[167]:


print(x_test.shape)


# In[168]:


# The known number of output classes.
num_classes = 2

# Input image dimensions
input_shape = (4,)

# Convert class vectors to binary class matrices. This uses 1 hot encoding.
y_train_binary = keras.utils.to_categorical(y_train, num_classes)
y_test_binary = keras.utils.to_categorical(y_test, num_classes)
#x_train_binary = keras.utils.to_categorical(x_train, num_classes)
#x_test_binary = keras.utils.to_categorical(x_test, num_classes)



#x_train = preprocessing.normalize(x_train)
#x_test=preprocessing.normalize(x_test)

x_train = x_train.reshape(x_train.shape[0],40,1)
x_test = x_test.reshape(x_test.shape[0],40,1)


# In[ ]:





# In[169]:


mean = np.mean(x_train, axis=0)
std = np.std(x_train, axis=0)
maxi=np.max(x_train,axis=0)
maxiTEST=np.max(x_test,axis=0)
mini=np.min(x_train,axis=0)
miniTEST=np.min(x_test,axis=0)

vmax = 10000
vmin = 10


#Single Feature Scaling
#x_train = x_train/maxi
#x_test = x_test/maxiTEST

#Min Max Normalization

#x_train=(x_train-mini)/(maxi-mini)
#x_test=(x_test-miniTEST)/(maxiTEST-miniTEST)


#z-score normalization
x_train = (x_train - mean)/std
x_test = (x_test - mean)/std


#Clipping

#x_train=x_train.apply(lambda x: vmax if x > vmax else vmin if x < vmin else x)
#x_test=x_test.apply(lambda x: vmax if x > vmax else vmin if x < vmin else x)

#Normalize

# normalize the data attributes


# Check the dataset now 
#x_train[150:160]


# In[170]:


print(x_train.shape)


# In[171]:


print(np.any(np.isnan(x_train)))


# In[172]:


from __future__ import print_function    
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import GridSearchCV


from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dropout, MaxPooling1D, Activation, BatchNormalization, Dense, Flatten, Conv1D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import TensorBoard


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[173]:


model = Sequential()
model.add(Conv1D(256, 8, padding='same',input_shape=(40,1)))  # X_train.shape[1] = No. of Columns
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv1D(128, 8, padding='same'))
#model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(Dropout(0.25))
model.add(MaxPooling1D(pool_size=(2), strides=1))
model.add(Conv1D(64, 8, padding='same'))
#model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(Dropout(0.25))
model.add(MaxPooling1D(pool_size=(4), strides=1))
model.add(Conv1D(64, 8, padding='same'))
model.add(Activation('relu'))
#model.add(Conv1D(64, 8, padding='same'))
#model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(2)) # Target class number
model.add(Activation('softmax'))
model.add(Dropout(0.25))
#opt = keras.optimizers.SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=False)
opt = keras.optimizers.Adam(lr=0.00001)
#opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
#model.summary()


# In[ ]:





# In[174]:


model.compile(loss='binary_crossentropy', optimizer=opt ,metrics=['accuracy'])
#model.summary()


# In[ ]:





# In[175]:


#print(x_test.shape)
#print(y_test_binary.shape)
model.summary()


# In[176]:


#print(y_test)


# In[ ]:





# In[177]:


batch_size = 64
epochs = 170
history=model.fit(x_train, y_train_binary,
          batch_size=batch_size,
          epochs=epochs,
          shuffle=False,
          verbose=1,
          validation_split=0.1)


# In[ ]:





# In[178]:


predictions = model.predict(x_test)
#print(predictions)
preds = model.evaluate(x = x_test,y = y_test_binary)
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))


# In[179]:


from sklearn.metrics import confusion_matrix



conf_matrix=confusion_matrix(np.argmax(y_test_binary,axis=1),np.argmax(predictions,axis=1))


#confusion_matrix = confusion_matrix(y_test_binary.argmax(axis=1), predictions.argmax(axis=1))
#conf_matrix = tf.math.confusion_matrix(labels=y_test_binary,
                                       #predictions=predictions)
print(conf_matrix)


# In[180]:


plt.rcParams.update({'font.size': 18})
plt.rcParams['axes.linewidth'] = 2.0

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'],color="red")
plt.title('Training Loss vs Validation Loss',weight='bold')
plt.ylabel('Loss',weight='bold')
plt.xlabel('epoch',weight='bold')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show()


# In[181]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'],color="red")
plt.title('Training Accuracy Vs. Validation Accuracy',weight='bold')
plt.ylabel('Accuracy',weight='bold')
plt.xlabel('epoch',weight='bold')
plt.legend(['Training', 'validation'], loc='lower right')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




