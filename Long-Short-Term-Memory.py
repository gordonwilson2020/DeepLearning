######################################################
# Deep Learning Regression with Python               #
# Long Short-Term Memory                             #
# (c) Diego Fernandez Garcia 2015-2018               #
# www.exfinsis.com                                   #
######################################################

# 1. Packages Importing
import numpy as np
import pandas as pd
# import pandas_datareader.data as web
import sklearn.preprocessing as pr
import sklearn.decomposition as fe
import sklearn.model_selection as cv
import sklearn.metrics as fa
import matplotlib.pyplot as plt
import keras.models as nn
import keras.layers as ml
import keras.optimizers as opt
import keras.wrappers.scikit_learn as ks
import time

#########

# 2. Data Downloading or Reading

# Yahoo Finance
# data = web.DataReader('SPY', data_source='yahoo', start='2007-01-01', end='2016-01-01')
# spy = data['Adj Close']
# spy.columns = ['SPY.Adjusted']

# Data Reading
spy = pd.read_csv('Data//Deep-Learning-Regression-Data.txt', index_col='Date', parse_dates=True)

#########

# 3. Feature Creation

# 3.1. Target Feature
rspy = spy/spy.shift(1)-1
rspy.columns = ['rspy']

# 3.2. Predictor Features
rspy1 = rspy.shift(1)
rspy1.columns = ['rspy1']
rspy2 = rspy.shift(2)
rspy2.columns = ['rspy2']
rspy3 = rspy.shift(3)
rspy3.columns = ['rspy3']
rspy4 = rspy.shift(4)
rspy4.columns = ['rspy4']
rspy5 = rspy.shift(5)
rspy5.columns = ['rspy5']

# 3.3. All Features
rspyall = rspy
rspyall = rspyall.join(rspy1)
rspyall = rspyall.join(rspy2)
rspyall = rspyall.join(rspy3)
rspyall = rspyall.join(rspy4)
rspyall = rspyall.join(rspy5)
rspyall = rspyall.dropna()

# 3.4. Range Delimiting

# 3.4.1. Training Range
rspyt = rspyall['2007-01-01':'2014-01-01']

# 3.4.2. Testing Range
rspyf = rspyall['2014-01-01':'2016-01-01']

#########

# 4. Long Short-Term Memory Recurrent Neural Network

# 4.1. LSTM Regression Features

# 4.1.1. Features Pre-processing
# re-scaling range [0,1]
# prspyt = (rspyt-min(rspyall))/(max(rspyall)-min(rspyall))
scalet = pr.MinMaxScaler(feature_range=(0, 1)).fit(rspyall)
prspyt = scalet.transform(rspyt)
prspyf = scalet.transform(rspyf)

# 4.1.2. Target Feature
yt = prspyt[:, 0]
yf = prspyf[:, 0]

# 4.1.3. Predictor Features Selection
xta = prspyt[:, [1, 2, 5]]
xfa = prspyf[:, [1, 2, 5]]

# 4.1.4. Predictor Features Extraction
# Principal Components Analysis
xtb = fe.PCA().fit_transform(prspyt[:, 1:6])
xfb = fe.PCA().fit_transform(prspyf[:, 1:6])

# 4.2. LSTM Regression Model
def lstmt_model(lrate, inshape, lstmunits):
    # 4.2.1. Model Type
    lstmt = nn.Sequential()
    # 4.2.2. Model Layers
    # Number of Hidden Layers
    # SimpleRNN: units: number of output/input features, input_shape: 3D data dimension (samples, time-steps, features)
    # Dense: units: number of output features
    # Activation: activation function options: 'linear' activation function
    # Dropout: nodes regularization rate, ActivityRegularization: nodes weights l1 or l2 norm regularization
    # Recurrent Hidden Layer 1
    lstmt.add(ml.Reshape(inshape + (1,), input_shape=inshape))
    lstmt.add(ml.LSTM(units=lstmunits))
    lstmt.add(ml.Activation(activation='linear'))
    lstmt.add(ml.Dropout(rate=0.0))
    lstmt.add(ml.ActivityRegularization(l1=0.0, l2=0.0))
    # Hidden Layer 1
    lstmt.add(ml.Dense(units=1))
    lstmt.add(ml.Activation(activation='linear'))
    lstmt.add(ml.Dropout(rate=0.0))
    lstmt.add(ml.ActivityRegularization(l1=0.0, l2=0.0))
    # 4.2.3. Model Optimization Algorithm
    # Back-propagation optimization options: 'SGD' stochastic gradient descent
    # Optimization regularization: learning rate 'lr', 'momentum' and learning rate 'decay'
    lstmoptt = opt.SGD(lr=lrate, momentum=0, decay=0, nesterov=False)
    # 4.2.4. Model Compilation
    # Loss function options: 'mean_squared_error'
    lstmt.compile(loss='mean_squared_error', optimizer=lstmoptt, metrics=['accuracy'])
    return lstmt

# 4.3. LSTM Regression Training Optimal Parameter Selection

# 4.3.1. Time Series Cross-Validation
# Exhaustive Grid Search Time Series Cross-Validation with Parameter Array Specification
# TimeSeriesSplit = anchored time series cross-validation with
# initial training subset = validating subset ~ number of observations per sample / (number of splits + 1) in size
lstmtsa = time.time()
cvlstmta = cv.GridSearchCV(ks.KerasRegressor(lstmt_model), cv=cv.TimeSeriesSplit(n_splits=5),
                          param_grid={"lrate": [0.01, 0.10], "inshape": [(3,)], "lstmunits": [3]}).fit(xta, yt)
lstmtea = time.time()
lstmtsb = time.time()
cvlstmtb = cv.GridSearchCV(ks.KerasRegressor(lstmt_model), cv=cv.TimeSeriesSplit(n_splits=5),
                          param_grid={"lrate": [0.01, 0.10], "inshape": [(5,)], "lstmunits": [5]}).fit(xtb, yt)
lstmteb = time.time()

# 4.3.2. Time Series Cross-Validation Optimal Parameter Selection
cvlstmpara = cvlstmta.best_params_
cvlstmparb = cvlstmtb.best_params_
print("")
print("== LSTM Regression Training Optimal Parameter Selection ==")
print("")
print("LSTM Regression A Optimal Learning Rate: ", cvlstmpara)
print("LSTM Regression B Optimal Learning Rate: ", cvlstmparb)
print("")
print("LSTM Regression A Training Time: ", (lstmtea-lstmtsa)/60, " minutes")
print("LSTM Regression B Training Time: ", (lstmteb-lstmtsb)/60, " minutes")
print("")

# 4.4. LSTM Algorithm Training
lstmta = cvlstmta.best_estimator_.model
lstmtb = cvlstmtb.best_estimator_.model

# 4.5. LSTM Regression Testing
# re-scaling back to original scale, values.reshape(-1,1) for single feature, values.reshape(1,-1) for single sample
lstmfa = lstmta.predict(x=xfa)
lstmfb = lstmtb.predict(x=xfb)
scalef = pr.MinMaxScaler(feature_range=(0, 1)).fit(rspyall['rspy'].values.reshape(-1, 1))
lstmfa = scalef.inverse_transform(lstmfa)
lstmfb = scalef.inverse_transform(lstmfb)

# 4.5.1. Testing Charts

# LSTM Regression Testing Chart A
rnnfadf = pd.DataFrame(lstmfa, index=rspyf.index)
fig1, ax = plt.subplots()
ax.plot(rspyf['rspy'])
ax.plot(rnnfadf, label='lstmfa')
plt.legend(loc='upper left')
plt.title('LSTM Regression Testing Chart A')
plt.ylabel('Arithmetic Returns')
plt.xlabel('Date')
plt.show()

# LSTM Regression Testing Chart B
lstmfbdf = pd.DataFrame(lstmfb, index=rspyf.index)
fig2, ax = plt.subplots()
ax.plot(rspyf['rspy'])
ax.plot(lstmfbdf, label='lstmfb')
plt.legend(loc='upper left')
plt.title('LSTM Regression Testing Chart B')
plt.ylabel('Arithmetic Returns')
plt.xlabel('Date')
plt.show()

# 4.6. LSTM Regression Testing Forecasting Accuracy
lstmmaea = fa.mean_absolute_error(rspyf['rspy'], lstmfa)
lstmmaeb = fa.mean_absolute_error(rspyf['rspy'], lstmfb)
lstmmsea = fa.mean_squared_error(rspyf['rspy'], lstmfa)
lstmmseb = fa.mean_squared_error(rspyf['rspy'], lstmfb)
lstmrmsea = np.sqrt(lstmmsea)
lstmrmseb = np.sqrt(lstmmseb)
print("== LSTM Regression Testing Forecasting Accuracy ==")
print("")
print("Mean Absolute Error ", "A:", round(lstmmaea, 6), "B:", round(lstmmaeb, 6))
print("Mean Squared Error ", "A:", round(lstmmsea, 6), "B:", round(lstmmseb, 6))
print("Root Mean Squared Error ", "A:", round(lstmrmsea, 6), "B:", round(lstmrmseb, 6))