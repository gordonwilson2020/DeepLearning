######################################################
# Deep Learning Regression with Python               #
# Deep Neural Network                                #
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
from tensorflow import optimizers as opt
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

# 4. Deep Neural Network

# 4.1. DNN Regression Features

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

# 4.2. DNN Regression Model
def dnnt_model(drate, inshape):
    # 4.2.1. Model Type
    dnnt = nn.Sequential()
    # 4.2.2. Model Layers
    # Number of Hidden Layers
    # Dense: units: number of output features, input_shape: 2D data dimension (samples and features)
    # Activation: activation function options: 'linear' activation function
    # Dropout: nodes regularization rate, ActivityRegularization: nodes weights l1 or l2 norm regularization
    # Hidden Layer 1
    dnnt.add(ml.Dense(units=1, input_shape=inshape))
    dnnt.add(ml.Activation(activation='linear'))
    dnnt.add(ml.Dropout(rate=drate))
    dnnt.add(ml.ActivityRegularization(l1=0.0, l2=0.0))
    # Hidden Layer 2
    dnnt.add(ml.Dense(units=1))
    dnnt.add(ml.Activation(activation='linear'))
    dnnt.add(ml.Dropout(rate=drate))
    dnnt.add(ml.ActivityRegularization(l1=0.0, l2=0.0))
    # Hidden Layer 3
    dnnt.add(ml.Dense(units=1))
    dnnt.add(ml.Activation(activation='linear'))
    dnnt.add(ml.Dropout(rate=drate))
    dnnt.add(ml.ActivityRegularization(l1=0.0, l2=0.0))
    # 4.2.3. Model Optimization Algorithm
    # Back-propagation optimization options: 'SGD' stochastic gradient descent
    # Optimization regularization: learning rate 'lr', 'momentum' and learning rate 'decay'
    dnnoptt = opt.SGD(learning_rate=0.01, momentum=0, decay=0, nesterov=False)
    # 4.2.4. Model Compilation
    # Loss function options: 'mean_squared_error'
    dnnt.compile(loss='mean_squared_error', optimizer=dnnoptt, metrics=['accuracy'])
    return dnnt

# 4.3. DNN Regression Training Optimal Parameter Selection

# 4.3.1. Time Series Cross-Validation
# Exhaustive Grid Search Time Series Cross-Validation with Parameter Array Specification
# TimeSeriesSplit = anchored time series cross-validation with
# initial training subset = validating subset ~ number of observations per sample / (number of splits + 1) in size
dnntsa = time.time()
cvdnnta = cv.GridSearchCV(ks.KerasRegressor(dnnt_model), cv=cv.TimeSeriesSplit(n_splits=5),
                          param_grid={"drate": [0.00, 0.25], "inshape": [(3,)]}).fit(xta, yt)
dnntea = time.time()
dnntsb = time.time()
cvdnntb = cv.GridSearchCV(ks.KerasRegressor(dnnt_model), cv=cv.TimeSeriesSplit(n_splits=5),
                          param_grid={"drate": [0.00, 0.25], "inshape": [(5,)]}).fit(xtb, yt)
dnnteb = time.time()

# 4.3.2. Time Series Cross-Validation Optimal Parameter Selection
cvdnnpara = cvdnnta.best_params_
cvdnnparb = cvdnntb.best_params_
print("")
print("== DNN Regression Training Optimal Parameter Selection ==")
print("")
print("DNN Regression A Optimal Dropout Rate: ", cvdnnpara)
print("DNN Regression B Optimal Dropout Rate: ", cvdnnparb)
print("")
print("DNN Regression A Training Time: ", (dnntea-dnntsa)/60, " minutes")
print("DNN Regression B Training Time: ", (dnnteb-dnntsb)/60, " minutes")
print("")

# 4.4. DNN Algorithm Training
dnnta = cvdnnta.best_estimator_.model
dnntb = cvdnntb.best_estimator_.model

# 4.5. DNN Regression Testing
# re-scaling back to original scale, values.reshape(-1,1) for single feature, values.reshape(1,-1) for single sample
dnnfa = dnnta.predict(x=xfa)
dnnfb = dnntb.predict(x=xfb)
scalef = pr.MinMaxScaler(feature_range=(0, 1)).fit(rspyall['rspy'].values.reshape(-1, 1))
dnnfa = scalef.inverse_transform(dnnfa)
dnnfb = scalef.inverse_transform(dnnfb)

# 4.5.1. Testing Charts

# DNN Regression Testing Chart A
dnnfadf = pd.DataFrame(dnnfa, index=rspyf.index)
fig1, ax = plt.subplots()
ax.plot(rspyf['rspy'])
ax.plot(dnnfadf, label='dnnfa')
plt.legend(loc='upper left')
plt.title('DNN Regression Testing Chart A')
plt.ylabel('Arithmetic Returns')
plt.xlabel('Date')
plt.show()

# DNN Regression Testing Chart B
dnnfbdf = pd.DataFrame(dnnfb, index=rspyf.index)
fig2, ax = plt.subplots()
ax.plot(rspyf['rspy'])
ax.plot(dnnfbdf, label='dnnfb')
plt.legend(loc='upper left')
plt.title('DNN Regression Testing Chart B')
plt.ylabel('Arithmetic Returns')
plt.xlabel('Date')
plt.show()

# 4.6. DNN Regression Testing Forecasting Accuracy
dnnmaea = fa.mean_absolute_error(rspyf['rspy'], dnnfa)
dnnmaeb = fa.mean_absolute_error(rspyf['rspy'], dnnfb)
dnnmsea = fa.mean_squared_error(rspyf['rspy'], dnnfa)
dnnmseb = fa.mean_squared_error(rspyf['rspy'], dnnfb)
dnnrmsea = np.sqrt(dnnmsea)
dnnrmseb = np.sqrt(dnnmseb)
print("== DNN Regression Testing Forecasting Accuracy ==")
print("")
print("Mean Absolute Error ", "A:", round(dnnmaea, 6), "B:", round(dnnmaeb, 6))
print("Mean Squared Error ", "A:", round(dnnmsea, 6), "B:", round(dnnmseb, 6))
print("Root Mean Squared Error ", "A:", round(dnnrmsea, 6), "B:", round(dnnrmseb, 6))