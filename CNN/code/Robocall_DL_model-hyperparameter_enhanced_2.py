#!/usr/bin/env python
# coding: utf-8

# In[57]:


import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
df = pd.read_csv('robocall_enhanced_2.csv', header=None, names=['smean', 'smax', 'smin', 'std_dev', 'variance','range','cv','skewness','kurtosis','q25','q50','q75','mean_crossing_rate','power','entropy','frequency_ratio','irrk','irrj','sharpness','smoothness','specCentroid','specstddev','specCrest','specSkewness','specKurt','maxfreq','numPeaks','zeroCrossingRate','slopeChanges','numInflectionPoint','totalPeak','silentTime','d1','d2','d3','d4','d5','d6','d7','d8','d9','d10','class'])


# In[34]:


df.tail()


# In[35]:


df['labels'] =df['class'].astype('category').cat.codes


# In[36]:


df=df.drop(['sharpness'], axis='columns')


# In[ ]:





# In[37]:


df.tail()


# Start preprocess

# Preprocessing Test

# In[38]:


#df['smean'] = df['smean']/df['smean'].max()
#df['smean'] = (df['smean'] - df['smean'].min())/(df['smean'].max() - df['smean'].min())
#df['smoothness'] = (df['smoothness'] - df['smoothness'].min())/(df['smoothness'].max() - df['smoothness'].min())
#df['smax'] = (df['smax']-df['smax'].mean())/df['smax'].std()


# In[39]:


df.head()


# In[ ]:





# In[40]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from tensorflow import keras


# In[41]:


#Some Preprocessing TEST
#clipping test


# In[42]:


X = df[['smean', 'smax', 'smin', 'std_dev', 'variance','range','cv','skewness','kurtosis','q25','q50','q75','mean_crossing_rate','power','entropy','frequency_ratio','irrk','irrj','smoothness','specCentroid','specstddev','specCrest','specSkewness','specKurt','numPeaks','zeroCrossingRate','slopeChanges','numInflectionPoint','totalPeak','silentTime','d1','d2','d3','d4','d5','d6','d7','d8','d9','d10','maxfreq']]
Y = df['labels']
x_train, x_test, y_train, y_test = train_test_split(np.asarray(X), np.asarray(Y), test_size=0.2, shuffle= True)


# In[43]:


print(x_test.shape)


# In[44]:


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

x_train = x_train.reshape(x_train.shape[0],41,1)
x_test = x_test.reshape(x_test.shape[0],41,1)


# In[45]:


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


# In[46]:


print(x_train.shape)


# In[47]:


print(np.any(np.isnan(x_train)))


# In[48]:


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




# In[49]:


import hyperopt
from hyperopt import fmin, tpe, hp
from tqdm import tqdm


# In[50]:


# Define the search space
space = {
    'batch_size': hp.choice('batch_size', [32, 64, 128]),
    'learning_rate': hp.loguniform('learning_rate', -5, -2),  # Log-uniform search for learning rate
    'filters': hp.choice('filters', [128, 256, 512]),
    'conv_layers': hp.choice('conv_layers', [1, 2, 3, 4]),
    'fc_layers': hp.choice('fc_layers', [1, 2]),
    'conv_units': hp.choice('conv_units', [64, 128, 256]),
    'fc_units': hp.choice('fc_units', [64, 128, 256]),
}


# In[51]:


# Define the objective function
def objective(params):
    model = Sequential()

    # Add convolutional layers
    for i in range(params['conv_layers']):
        model.add(Conv1D(params['conv_units'], 8, padding='same', input_shape=(41, 1)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=(2)))

    model.add(Flatten())

    # Add fully connected layers
    for i in range(params['fc_layers']):
        model.add(Dense(params['fc_units']))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.25))

    model.add(Dense(2))
    model.add(Activation('softmax'))

    opt = Adam(lr=params['learning_rate'])
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    history = model.fit(x_train, y_train_binary, batch_size=params['batch_size'], epochs=70, shuffle=False, verbose=0, validation_split=0.1)
    
    # Minimize negative accuracy
    return -history.history['val_accuracy'][-1]


# In[52]:


# Run the optimization
num_evals = 20
progress_bar = tqdm(total=num_evals)

# Define a counter to track progress
counter = 0


# In[53]:


def update_progress(*args, **kwargs):
    global counter
    counter += 1
    progress_bar.update(1)


# In[54]:


# Run the optimization with the workaround
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=num_evals)
progress_bar.close()

print("Best Hyperparameters:", best)


# In[ ]:





# In[55]:


# Extract the best hyperparameters
best_batch_size = [32, 64, 128][best['batch_size']]
best_learning_rate = best['learning_rate']
best_filters = [128, 256, 512][best['filters']]
best_conv_layers = [1, 2, 3][best['conv_layers']]
best_fc_layers = [1, 2][best['fc_layers']]
best_conv_units = [64, 128, 256][best['conv_units']]
best_fc_units = [64, 128, 256][best['fc_units']]

# Train the model with the best hyperparameters
best_model = Sequential()

# Add convolutional layers
for i in range(best_conv_layers):
    best_model.add(Conv1D(best_conv_units, 8, padding='same', input_shape=(41, 1)))
    best_model.add(Activation('relu'))
    best_model.add(BatchNormalization())
    best_model.add(MaxPooling1D(pool_size=(2)))

best_model.add(Flatten())

# Add fully connected layers
for i in range(best_fc_layers):
    best_model.add(Dense(best_fc_units))
    best_model.add(Activation('relu'))
    best_model.add(BatchNormalization())
    best_model.add(Dropout(0.25))

best_model.add(Dense(2))
best_model.add(Activation('softmax'))

opt = Adam(lr=best_learning_rate)
best_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])


# In[56]:


# Train the model
best_history = best_model.fit(x_train, y_train_binary, batch_size=best_batch_size, epochs=70, shuffle=False, verbose=1, validation_split=0.1)


# In[31]:


# Evaluate the model
predictions = best_model.predict(x_test)
preds = best_model.evaluate(x=x_test, y=y_test_binary)

print()
print("Loss =", preds[0])
print("Test Accuracy =", preds[1])


# In[32]:


from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(np.argmax(y_test_binary, axis=1), np.argmax(predictions, axis=1))
print(conf_matrix)


# In[40]:


# Plot training history
plt.rcParams.update({'font.size': 18})
plt.rcParams['axes.linewidth'] = 2.0


# In[41]:


plt.plot(best_history.history['loss'])
plt.plot(best_history.history['val_loss'], color="red")
plt.title('Training Loss vs Validation Loss', weight='bold')
plt.ylabel('Loss', weight='bold')
plt.xlabel('epoch', weight='bold')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show()


# In[ ]:





# In[105]:


plt.plot(best_history.history['accuracy'])
plt.plot(best_history.history['val_accuracy'], color="red")
plt.title('Training Accuracy Vs. Validation Accuracy', weight='bold')
plt.ylabel('Accuracy', weight='bold')
plt.xlabel('epoch', weight='bold')
plt.legend(['Training', 'validation'], loc='lower right')
plt.show()


# In[ ]:





# In[ ]:




