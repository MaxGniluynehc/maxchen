
# Gradient Norm Scaling/Clipping
from keras import optimizers
# configure sgd with gradient norm scaling
# i.e. changing the derivatives of the loss function to have a given vector norm when
# the L2 vector norm (sum of the squared values) of the gradient vector exceeds
# a threshold value.
opt = optimizers.SGD(lr=0.01, momentum=0.9, clipnorm=1.0)


# configure sgd with gradient norm clipping
# clipping the derivatives of the loss function to have a given value if a gradient value is less
# than a negative threshold or more than the positive threshold.
opt = optimizers.SGD(lr=0.01, momentum=0.9, clipvalue=1.0)

#######################################################################




# regression predictive modeling problem
from sklearn.datasets import make_regression
from matplotlib import pyplot
# generate regression dataset
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=1)
# histogram of target variable
pyplot.subplot(131)
pyplot.hist(y)
# boxplot of target variable
pyplot.subplot(132)
pyplot.boxplot(y)
pyplot.show()
# scatter plot
pyplot.subplot(133)
pyplot.show(X,y)


####################################################################

# mlp with unscaled data for the regression problem
from sklearn.datasets import make_regression
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from matplotlib import pyplot
# generate regression dataset
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=1)
# split into train and test
n_train = 500
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
# define model
model = Sequential()  # the model with a linear stack of layers
model.add(Dense(25, input_dim=20, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(1, activation='linear'))
# compile model
# model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01, momentum=0.9))
opt_scaling = SGD(lr=0.01, momentum=0.9, clipvalue=5.0)
model.compile(loss='mean_squared_error', optimizer=opt_scaling)

# fit model
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=100, verbose=0)
# evaluate the model
train_mse = model.evaluate(trainX, trainy, verbose=0)
test_mse = model.evaluate(testX, testy, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_mse, test_mse))
# plot loss during training
pyplot.title('Mean Squared Error')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# The model above is NOT able to learn for the problem, resulting in nans.

# Solutions:
# 1. The traditional solution is to rescale the target variable using either standardization or normalization.
# 2. using Gradient Norm Scaling: replace the optimizer with:
opt_scaling = optimizers.SGD(lr=0.01, momentum=0.9, clipnorm=1.0)
# 3. using Gradient Norm Clipping: replace the optimizer with:
opt_clipping = SGD(lr=0.01, momentum=0.9, clipvalue=5.0)



















