import wfdb
import numpy as np
import matplotlib.pyplot as plt
import os

import numpy
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Conv1D, Conv2D, MaxPooling2D, Flatten, ConvLSTM2D, BatchNormalization, Conv3D
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import scipy.io as sio
from os import listdir
from os.path import isfile, join
import numpy as np
import keras
from sklearn.metrics import accuracy_score
from keras import backend as K
import sys

#downloads data into directory
wfdb.dl_database('mitdb', os.path.join(os.getcwd(), 'mitdb'))

#sets path to data
data_path = "mitdb/"

#marks out patients
pts = ['100','101','102','103','104','105','106','107',
       '108','109','111','112','113','114','115','116',
       '117','118','119','121','122','123','124','200',
       '201','202','203','205','207','208','209','210',
       '212','213','214','215','217','219','220','221',
       '222','223','228','230','231','232','233','234']

#marks out data not relevant to heartbeats
nonbeat = ['[','!',']','x','(',')','p','t','u','`',
           '\'','^','|','~','+','s','T','*','D','=','"','@','Q','?']
abnormal = ['L','R','V','/','A','f','F','j','a','E','J','e','S']
df = pd.DataFrame()

#build df of patients and unique values
for pt in pts:
    file = data_path + pt
    annotation = wfdb.rdann(file, 'atr')
    sym = annotation.symbol

    values, counts = np.unique(sym, return_counts=True)
    df_sub = pd.DataFrame({'sym':values, 'val':counts, 'pt':[pt]*len(counts)})
    df = pd.concat([df, df_sub],axis = 0)

#anything abnormal is given a 1, else 0
df['cat'] = -1
df.loc[df.sym == 'N','cat'] = 0
df.loc[df.sym.isin(abnormal), 'cat'] = 1

#takes in path to a file and returns tuple w/ signal, annotation symbol, and annotation sample
def load_ecg(file):
    #load the ecg
    record = wfdb.rdrecord(file)
    annotation = wfdb.rdann(file, 'atr')

    p_signal = record.p_signal
    assert record.fs == 360, 'sample freq is not 360'

    atr_sym = annotation.symbol
    atr_sample = annotation.sample

    #remember is tuple
    return p_signal, atr_sym, atr_sample

#PLOTTING
#record = wfdb.rdrecord(data_path + '100')
#wfdb.plot_wfdb(record)

#takes in patients, time freq, 
def make_dataset(pts, num_sec, fs, abnormal):
    # function for making dataset ignoring non-beats
    # input:
    # pts - list of patients
    # num_sec = number of seconds to include before and after the beat
    # fs = frequency
    # output:
    #   X_all = signal (nbeats , num_sec * fs columns)
    #   Y_all = binary is abnormal (nbeats, 1)
    #   sym_all = beat annotation symbol (nbeats,1)

    #initialize numpy arrays
    num_cols = 2*num_sec * fs
    X_all = np.zeros((1,num_cols))
    Y_all = np.zeros((1,1))
    sym_all = []

    #track beats across patients
    max_rows = []
    i = 0
    for pt in pts:
        if (i > 1): break
        file = data_path + pt

        p_signal, atr_sym, atr_sample = load_ecg(file)
        p_signal = p_signal[:,0]

        #load df with beats only
        df_ann = pd.DataFrame({'atr_sym':atr_sym,
                              'atr_sample':atr_sample})
        df_ann = df_ann.loc[df_ann.atr_sym.isin(abnormal + ['N'])]

        X,Y,sym = build_XY(p_signal,df_ann, num_cols, abnormal)
        sym_all = sym_all+sym
        max_rows.append(X.shape[0])
        X_all = np.append(X_all,X,axis = 0)
        Y_all = np.append(Y_all,Y,axis = 0)

    #drop the first zero row
    X_all = X_all[1:,:]
    Y_all = Y_all[1:,:]
    print(X_all.shape)

    #assert dimensions 
    assert np.sum(max_rows) == X_all.shape[0], 'number of X, max_rows rows messed up'
    assert Y_all.shape[0] == X_all.shape[0], 'number of X, Y rows messed up'
    assert Y_all.shape[0] == len(sym_all), 'number of Y, sym rows messed up'

    return X_all, Y_all, sym_all

#builds X and y set to be split later
def build_XY(p_signal, df_ann, num_cols, abnormal):

    num_rows = len(df_ann)
    X = np.zeros((num_rows, num_cols))
    Y = np.zeros((num_rows,1))
    sym = []


    row = 0

    for atr_sample, atr_sym in zip(df_ann.atr_sample.values,df_ann.atr_sym.values):
        #get boundaries
        left = max([0,(atr_sample - num_sec*fs) ])
        right = min([len(p_signal),(atr_sample + num_sec*fs) ])
        x = p_signal[left: right]
        #add to X,y if verified properly
        if len(x) == num_cols:
            X[row,:] = x
            Y[row,:] = int(atr_sym in abnormal)
            sym.append(atr_sym)
            row += 1
    X = X[:row,:]
    Y = Y[:row,:]
    return X,Y,sym

num_sec = 3
fs = 360
X_all, Y_all, sym_all = make_dataset(pts, num_sec, fs, abnormal)


#split built X and y into train and test
X_train, X_valid, y_train, y_valid = train_test_split(X_all, Y_all, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25, random_state=42)


#reframing dimensions of the data
n = 20
m = 108
t = 1
c = 1
X_train = numpy.reshape(X_train, (X_train.shape[0], t, n, m, c))
X_valid = numpy.reshape(X_valid, (X_valid.shape[0], t, n, m, c))
# X_test = numpy.reshape(X_test, (X_test.shape[0], t, n, m, c))
image_size = (t, n, m, c)

#instantiate and modify keras model
batch_size = 32
model = Sequential()

model.add(ConvLSTM2D(64, (3, 3), activation='relu', input_shape=image_size, return_sequences=True, padding='same'))
model.add(BatchNormalization())

model.add(ConvLSTM2D(128, (3, 3), activation='relu', return_sequences=True, padding='same'))
model.add(BatchNormalization())

model.add(Conv3D(64, (3, 3, 3), activation='relu', padding='same'))

model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'Recall'])

#epoch is adjustable (99.9 auc at 6)
model.fit(X_train, y_train, epochs=1, batch_size=batch_size, validation_data=(X_valid, y_valid), verbose=1, shuffle=False)
model.save('Model/my_model2.h5')

X_test = numpy.reshape(X_test, (X_test.shape[0], t, n, m, c))
pred = model.predict(X_test,verbose = 1)
print(pred)
print(y_test)

X_test_flattened = X_test.reshape(X_test.shape[0], -1) 
if not os.path.exists('data'):
    os.makedirs('data')

df = pd.DataFrame(X_test_flattened)
df.to_csv('data/X_test.csv')

df = pd.DataFrame(y_test)
df.to_csv('data/y_test.csv')

y_train_preds = model.predict(X_train,verbose = 1)
y_valid_preds = model.predict(X_valid,verbose = 1)


#print accuracy metrics
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
def calc_prevalence(y_actual):
    return (sum(y_actual)/len(y_actual))

  
    return sum((y_pred < thresh) & (y_actual == 0)) /sum(y_actual ==0)
def print_report(y_actual, y_pred, thresh):

    auc = roc_auc_score(y_actual, y_pred)
    accuracy = accuracy_score(y_actual, (y_pred > thresh))
    recall = recall_score(y_actual, (y_pred > thresh))
    precision = precision_score(y_actual, (y_pred > thresh))
    print('AUC:%.3f'%auc)
    print('accuracy:%.3f'%accuracy)
    print('recall:%.3f'%recall)
    print('precision:%.3f'%precision)
    print(' ')
    return auc, accuracy, recall, precision

thresh = (sum(y_train)/len(y_train))[0]
print('Train');
print_report(y_train, y_train_preds, thresh)
print('Valid');
print_report(y_valid, y_valid_preds, thresh);