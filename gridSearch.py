import numpy
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
import time
from keras.models import Sequential
from keras.layers import Dense, Dropout, AlphaDropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm

from prepareDATA import *

'''
XDev = trainNN.XDev 
YDev = trainNN.YDev 
'''
compileArgs = {'loss': 'binary_crossentropy', 'optimizer': 'adam', 'metrics': ["accuracy"]}
trainParams = {'epochs': 2, 'batch_size': 10, 'verbose': 1}

# Fix seed for reproducibility
seed = 42
numpy.random.seed(seed)

# Tune the Number of Neurons in the Hidden Layer 
def myClassifier(nIn=len(trainFeatures), nOut=1, compileArgs=compileArgs, layers=1, neurons =1):
    model = Sequential()
    model.add(Dense(nIn, input_dim=nIn, kernel_initializer='he_normal', activation='relu'))
    for i in range(0,layers):
        model.add(Dense(neurons, kernel_initializer='he_normal', activation='relu'))
    model.add(Dense(nOut, activation="sigmoid", kernel_initializer='glorot_normal'))
    model.compile(**compileArgs)
    print("Training with %i layers and %i neurons" % (layers, neurons))
    return model


model = KerasClassifier(build_fn=myClassifier, epochs = 2, batch_size = 20, verbose = 1)

neurons = [10, 12, 14]
layers = [1,2]
#layers = [1,2,3,4,5]
param_grid = dict(neurons=neurons, layers=layers)
grid = GridSearchCV(estimator = model, param_grid = param_grid, n_jobs=-1) #n_jobs = -1 -> Total number of CPU/GPU cores
print("Starting the training")
start = time.time()
grid_result = grid.fit(XDev,YDev)
print("Training took ", time.time()-start, " seconds")

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
