import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.4f' % x)
import seaborn as sns
sns.set_context("paper", font_scale=1.3)
sns.set_style('white')
import warnings
warnings.filterwarnings('ignore')
from time import time
import matplotlib.ticker as tkr
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from sklearn import preprocessing
from statsmodels.tsa.stattools import pacf
%matplotlib inline
import math
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.callbacks import EarlyStopping
from keras.models import load_model
# import modules and read csv file
# I have some problem with work on github. Github doesnt open ipynb file and I decided create new file on github without upload him. 
# I think that you should just copy code in your file on jupiter notebook, I make indentation (the borders copy)
df = pd.read_csv('bitcoin_csv.csv')
df = df.dropna()[['price(USD)','date']].rename({'price(USD)':'pr'}, axis=1)
df.tail(1)
"-----------------------"
df.date = pd.to_datetime(df.date)
# date to datetime.
dataset2 = df.pr.values #numpy.ndarray
dataset1 = dataset2.astype('float32') # change type to float
dataset = np.reshape(dataset1, (-1, 1)) 
scaler = MinMaxScaler(feature_range=(0, 1)) # normalize for input model
dataset = scaler.fit_transform(dataset) 
train_size = int(len(dataset) * 0.80) # make train size and test_size
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:] #make train and test dataset
"------------------"
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)
    
look_back = 35 # 
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
We store 35 prices in a multidimensional array (X_train). 
Each 35 value in the array is a prediction for the next array. 
Y_train is the actual result that should result 
from the prediction. You can look at X_train and Y_train 
and figure it out yourself again
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"---------------------------------------------"
X_train, Y_train = create_dataset(train, look_back)
X_test, Y_test = create_dataset(test, look_back)
print(X_train)
print(Y_train)
"------------------------------------------------"
import keras # import keras and build model
model = Sequential()
model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2]))) #RNN 
model.add(Dropout(0.2)) 
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

history = model.fit(X_train, Y_train, epochs=1000, batch_size=70, validation_data=(X_test, Y_test), 
                    callbacks=[keras.callbacks.ModelCheckpoint('we1u.h5', monitor='loss', verbose=1, save_best_only=True, mode='min')])
#EarlyStopping(monitor='val_loss', patience=30)
model.summary()
model = load_model('we1u.h5')
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)
# invert predictions
train_predict = scaler.inverse_transform(train_predict)
Y_train = scaler.inverse_transform([Y_train])
test_predict = scaler.inverse_transform(test_predict)
Y_test = scaler.inverse_transform([Y_test])
print('Train Mean Absolute Error:', mean_absolute_error(Y_train[0], train_predict[:,0]))
print('Train Root Mean Squared Error:',np.sqrt(mean_squared_error(Y_train[0], train_predict[:,0])))
print('Test Mean Absolute Error:', mean_absolute_error(Y_test[0], test_predict[:,0]))
print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(Y_test[0], test_predict[:,0])))
aa=[x for x in range(376)]
plt.figure(figsize=(8,4))
plt.plot(aa, Y_test[0][:376], marker='.', label="actual")
plt.plot(aa, test_predict[:,0][:376], 'r', label="prediction")
# plt.tick_params(left=False, labelleft=True) #remove ticks
plt.tight_layout()
sns.despine(top=True)
plt.subplots_adjust(left=0.07)
plt.ylabel('Prices', size=15)
plt.xlabel('Time step', size=15)
plt.legend(fontsize=15)
plt.show();
