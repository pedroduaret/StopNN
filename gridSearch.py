'''
Code inspired from: https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys
import numpy
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
import time
from keras.models import Sequential
from keras.layers import Dense, Dropout, AlphaDropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
from prepareDATA import *
import localConfig as cfg
import datetime 

os.chdir(cfg.lgbk+"gridSearches")

compileArgs = {'loss': 'binary_crossentropy', 'optimizer': 'adam', 'metrics': ["accuracy"]}

# Fix seed for reproducibility
seed = 42
numpy.random.seed(seed)

# Tune the Number of Neurons in the Hidden Layer 
def myClassifier(nIn=len(trainFeatures), nOut=1, compileArgs=compileArgs, layers=1, neurons=1):
    model = Sequential()
    model.add(Dense(neurons, input_dim=nIn, kernel_initializer='he_normal', activation='relu'))
    for i in range(0,layers-1):
        model.add(Dense(neurons, kernel_initializer='he_normal', activation='relu'))
    model.add(Dense(nOut, activation="sigmoid", kernel_initializer='glorot_normal'))
    model.compile(**compileArgs)
    print("\nTraining with %i layers and %i neurons\n" % (layers, neurons))
    return model


model = KerasClassifier(build_fn=myClassifier,batch_size=20, verbose = 1)

#Hyperparameters
neurons = [8,10,12,14]
layers = [1,2]
#batch_size = [10, 20, 40, 60, 80, 100]
epochs = [5, 10, 15]
learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]

now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
filename="gS:"+now+".txt"

param_grid = dict(neurons=neurons, layers=layers, epochs=epochs)#, batch_size=batch_size)
grid = GridSearchCV(estimator = model, param_grid = param_grid, n_jobs=3) #n_jobs = -1 -> Total number of CPU/GPU cores
print("Starting the training")
start = time.time()
grid_result = grid.fit(XDev,YDev)
sys.stdout=open(filename,"w")
print(now+"\n")

print("Training took ", time.time()-start, " seconds")

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
sys.stdout.close()
