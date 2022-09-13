######################################################
# Deep Learning Regression with Python               #
# Recurrent Neural Network                           #
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

# 4. Recurrent Neural Network

# 4.1. RNN Regression Features

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

# 4.2. RNN Regression Model
def rnnt_model(lrate, inshape, rnnunits):
    # 4.2.1. Model Type
    rnnt = nn.Sequential()
    # 4.2.2. Model Layers
    # Number of Hidden Layers
    # SimpleRNN: units: number of output/input features, input_shape: 3D data dimension (samples, time-steps, features)
    # Dense: units: number of output features
    # Activation: activation function options: 'linear' activation function
    # Dropout: nodes regularization rate, ActivityRegularization: nodes weights l1 or l2 norm regularization
    # Recurrent Hidden Layer 1
    rnnt.add(ml.Reshape(inshape + (1,), input_shape=inshape))
    rnnt.add(ml.SimpleRNN(units=rnnunits))
    rnnt.add(ml.Activation(activation='linear'))
    rnnt.add(ml.Dropout(rate=0.0))
    rnnt.add(ml.ActivityRegularization(l1=0.0, l2=0.0))
    # Hidden Layer 1
    rnnt.add(ml.Dense(units=1))
    rnnt.add(ml.Activation(activation='linear'))
    rnnt.add(ml.Dropout(rate=0.0))
    rnnt.add(ml.ActivityRegularization(l1=0.0, l2=0.0))
    # 4.2.3. Model Optimization Algorithm
    # Back-propagation optimization options: 'SGD' stochastic gradient descent
    # Optimization regularization: learning rate 'lr', 'momentum' and learning rate 'decay'
    rnnoptt = opt.SGD(lr=lrate, momentum=0, decay=0, nesterov=False)
    # 4.2.4. Model Compilation
    # Loss function options: 'mean_squared_error'
    rnnt.compile(loss='mean_squared_error', optimizer=rnnoptt, metrics=['accuracy'])
    return rnnt

# 4.3. RNN Regression Training Optimal Parameter Selection

# 4.3.1. Time Series Cross-Validation
# Exhaustive Grid Search Time Series Cross-Validation with Parameter Array Specification
# TimeSeriesSplit = anchored time series cross-validation with
# initial training subset = validating subset ~ number of observations per sample / (number of splits + 1) in size
rnntsa = time.time()
cvrnnta = cv.GridSearchCV(ks.KerasRegressor(rnnt_model), cv=cv.TimeSeriesSplit(n_splits=5),
                          param_grid={"lrate": [0.01, 0.10], "inshape": [(3,)], "rnnunits": [3]}).fit(xta, yt)
rnntea = time.time()
rnntsb = time.time()
cvrnntb = cv.GridSearchCV(ks.KerasRegressor(rnnt_model), cv=cv.TimeSeriesSplit(n_splits=5),
                          param_grid={"lrate": [0.01, 0.10], "inshape": [(5,)], "rnnunits": [5]}).fit(xtb, yt)
rnnteb = time.time()

# 4.3.2. Time Series Cross-Validation Optimal Parameter Selection
cvrnnpara = cvrnnta.best_params_
cvrnnparb = cvrnntb.best_params_
print("")
print("== RNN Regression Training Optimal Parameter Selection ==")
print("")
print("RNN Regression A Optimal Learning Rate: ", cvrnnpara)
print("RNN Regression B Optimal Learning Rate: ", cvrnnparb)
print("")
print("RNN Regression A Training Time: ", (rnntea-rnntsa)/60, " minutes")
print("RNN Regression B Training Time: ", (rnnteb-rnntsb)/60, " minutes")
print("")

# 4.4. RNN Algorithm Training
rnnta = cvrnnta.best_estimator_.model
rnntb = cvrnntb.best_estimator_.model

# 4.5. RNN Regression Testing
# re-scaling back to original scale, values.reshape(-1,1) for single feature, values.reshape(1,-1) for single sample
rnnfa = rnnta.predict(x=xfa)
rnnfb = rnntb.predict(x=xfb)
scalef = pr.MinMaxScaler(feature_range=(0, 1)).fit(rspyall['rspy'].values.reshape(-1, 1))
rnnfa = scalef.inverse_transform(rnnfa)
rnnfb = scalef.inverse_transform(rnnfb)

# 4.5.1. Testing Charts

# RNN Regression Testing Chart A
rnnfadf = pd.DataFrame(rnnfa, index=rspyf.index)
fig1, ax = plt.subplots()
ax.plot(rspyf['rspy'])
ax.plot(rnnfadf, label='rnnfa')
plt.legend(loc='upper left')
plt.title('RNN Regression Testing Chart A')
plt.ylabel('Arithmetic Returns')
plt.xlabel('Date')
plt.show()

# RNN Regression Testing Chart B
rnnfbdf = pd.DataFrame(rnnfb, index=rspyf.index)
fig2, ax = plt.subplots()
ax.plot(rspyf['rspy'])
ax.plot(rnnfbdf, label='rnnfb')
plt.legend(loc='upper left')
plt.title('RNN Regression Testing Chart B')
plt.ylabel('Arithmetic Returns')
plt.xlabel('Date')
plt.show()

# 4.6. RNN Regression Testing Forecasting Accuracy
rnnmaea = fa.mean_absolute_error(rspyf['rspy'], rnnfa)
rnnmaeb = fa.mean_absolute_error(rspyf['rspy'], rnnfb)
rnnmsea = fa.mean_squared_error(rspyf['rspy'], rnnfa)
rnnmseb = fa.mean_squared_error(rspyf['rspy'], rnnfb)
rnnrmsea = np.sqrt(rnnmsea)
rnnrmseb = np.sqrt(rnnmseb)
print("== RNN Regression Testing Forecasting Accuracy ==")
print("")
print("Mean Absolute Error ", "A:", round(rnnmaea, 6), "B:", round(rnnmaeb, 6))
print("Mean Squared Error ", "A:", round(rnnmsea, 6), "B:", round(rnnmseb, 6))
print("Root Mean Squared Error ", "A:", round(rnnrmsea, 6), "B:", round(rnnrmseb, 6))