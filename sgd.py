## ANN MiniProject 2

from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame
from sklearn.metrics import mean_squared_error
from pandas import concat
from numpy import concatenate
from math import sqrt
from keras import layers
import datetime
from keras.models import Sequential
from keras.layers import Dense, Dropout
import time
import matplotlib

from matplotlib import patches
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import pyplot


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


dataset = read_csv("UberDemand.csv", index_col=17)
# get rid of unnamed column
dataset.drop(dataset.columns[dataset.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)

b = dataset.loc[:, ['pickups']].T
c = b.values.tolist()
a = c[0]

f = [[], [], [], []]
for i in range(len(a)):
    f[i % 4].append(a[i])

Bronx = f[0]
Brooklyn = f[1]
Manhattan = f[2]
Queens = f[3]

dataset = dataset.drop(columns=['borough', 'pickups'])
dataset = dataset.drop_duplicates()
dataset['Bronx'] = Bronx
dataset['Brooklyn'] = Brooklyn
dataset['Manhattan'] = Manhattan
dataset['Queens'] = Queens

values = dataset.values
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 6, 1)
# drop columns we don't want to predict
listOfDrops = []
for i in range(108,122):
    listOfDrops.append(i)

# print("list of drops ", listOfDrops)
reframed.drop(reframed.columns[listOfDrops], axis=1, inplace=True)
# print(reframed.columns)
# print(reframed)
# print(dataset.tail())




# split into train and test sets
values = reframed.values
n_train_hours = 5 * 30 * 24
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# split into input and outputs
train_X, train_y = train[:, :-4], train[:, -4:]
test_X, test_y = test[:, :-4], test[:, -4:]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


NumOfEpochs = 300
# design network
model = Sequential()
model.add(layers.LSTM(NumOfEpochs, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(4))
model.compile(loss='mae', optimizer='sgd',metrics=['accuracy'])
# fit network
start = time.time()
history = model.fit(train_X, train_y, epochs=NumOfEpochs, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
end = time.time()
# plot history

f = pyplot.figure()
pyplot.plot(history.history['acc'], label='acc')
pyplot.plot(history.history['val_acc'], label='val_acc')
pyplot.legend()
pyplot.show()
now = datetime.datetime.now()
f.savefig("%s.png"%(str(now.hour)+"_"+str(now.minute)+"_"+str(now.second)+"_acc"), bbox_inches='tight')

f1 = pyplot.figure()
pyplot.plot(history.history['loss'], label='train loss')
pyplot.plot(history.history['val_loss'], label='val loss')
pyplot.legend()
pyplot.show()
now = datetime.datetime.now()
f1.savefig("%s.png"%(str(now.hour)+"_"+str(now.minute)+"_"+str(now.second)+"_loss"), bbox_inches='tight')

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 94:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,:]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 4))
inv_y = concatenate((test_y, test_X[:, 94:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,:]
print(inv_y.shape)
f3 = pyplot.figure()
pyplot.subplot(4, 1, 1)
pyplot.plot(inv_y[:,0],label='y0')
pyplot.plot(inv_yhat[:,0],label='y_hat0')
pyplot.legend()

pyplot.subplot(4, 1, 2)
pyplot.plot(inv_y[:,1],label='y1')
pyplot.plot(inv_yhat[:,1],label='y_hat1')
pyplot.legend()

pyplot.subplot(4, 1, 3)
pyplot.plot(inv_y[:,2],label='y2')
pyplot.plot(inv_yhat[:,2],label='y_hat2')
pyplot.legend()

pyplot.subplot(4, 1, 4)
pyplot.plot(inv_y[:,3],label='y3')
pyplot.plot(inv_yhat[:,3],label='y_hat3')
pyplot.legend()

pyplot.show()
now = datetime.datetime.now()
f3.savefig("%s.png"%(str(now.hour)+"_"+str(now.minute)+"_"+str(now.second)+"_realVSpredicted"), bbox_inches='tight')

# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
print('Elapsed time for fitting: ', end-start)