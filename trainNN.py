import root_numpy
import numpy as np
import pandas
import keras
import time
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix, cohen_kappa_score

loc = "./"
treeName = "bdttree"
baseSigName = "T2DegStop_300_270"
bkgDatasets = [
                "Wjets_200to400",
                "Wjets_400to600",
                "Wjets_600to800",
              ]

myFeatures = ["Jet1Pt", "Met", "Njet", "LepPt", "LepEta", "LepChg", "HT", "NbLoose"]
inputBranches = list(myFeatures)
inputBranches.append("XS")
inputBranches.append("weight")
preselection = "(DPhiJet1Jet2 < 2.5 || Jet2Pt < 60) && (Met > 280) && (HT > 200) && (isTight == 1) && (Jet1Pt > 110)"
suffix = "_skimmed"

print "Loading datasets..."
sigDataDev = pandas.DataFrame(root_numpy.root2array(
                                                    loc + "/train/" + baseSigName + suffix + ".root",
                                                    treename=treeName,
                                                    selection=preselection,
                                                    branches=inputBranches
                                                    ))
sigDataVal = pandas.DataFrame(root_numpy.root2array(
                                                    loc + "/test/" + baseSigName + suffix + ".root",
                                                    treename=treeName,
                                                    selection=preselection,
                                                    branches=inputBranches
                                                    ))
bkgDataDev = None
bkgDataVal = None
for bkgName in bkgDatasets:
  if bkgDataDev is None:
    bkgDataDev = pandas.DataFrame(root_numpy.root2array(
                                                        loc + "/train/" + bkgName + suffix + ".root",
                                                        treename=treeName,
                                                        selection=preselection,
                                                        branches=inputBranches
                                                        ))
    bkgDataVal = pandas.DataFrame(root_numpy.root2array(
                                                        loc + "/test/" + bkgName + suffix + ".root",
                                                        treename=treeName,
                                                        selection=preselection,
                                                        branches=inputBranches
                                                        ))
  else:
    bkgDataDev = bkgDataDev.append(
                                   pandas.DataFrame(root_numpy.root2array(
                                                                          loc + "/train/" + bkgName + suffix + ".root",
                                                                          treename=treeName,
                                                                          selection=preselection,
                                                                          branches=inputBranches
                                                                          )),
                                   ignore_index=True
                                   )
    bkgDataVal = bkgDataVal.append(
                                   pandas.DataFrame(root_numpy.root2array(
                                                                          loc + "/test/" + bkgName + suffix + ".root",
                                                                          treename=treeName,
                                                                          selection=preselection,
                                                                          branches=inputBranches
                                                                          )),
                                   ignore_index=True
                                   )

sigDataDev["category"] = 1
sigDataVal["category"] = 1
bkgDataDev["category"] = 0
bkgDataVal["category"] = 0

# Calculate event weights
# The input files already have a branch called weight, which contains the per-event weights
# These precomputed weights have all scale factors applied. If we desire to not use the scale factors
# we should compute a new set of weights ourselves. Remember to repeat for all datasets.
################### Add computation here if wanted #######################################################
# After computing the weights, the total class has to be normalized.
sigDataDev.weight = sigDataDev.weight/sigDataDev.weight.sum()
sigDataVal.weight = sigDataVal.weight/sigDataVal.weight.sum()
bkgDataDev.weight = bkgDataDev.weight/bkgDataDev.weight.sum()
bkgDataVal.weight = bkgDataVal.weight/bkgDataVal.weight.sum()


data = sigDataDev.copy()
data = data.append(sigDataVal.copy(), ignore_index=True)
data = data.append(bkgDataDev.copy(), ignore_index=True)
data = data.append(bkgDataVal.copy(), ignore_index=True)
print 'Datasets contain a total of', len(data), 'events:'
print '  Signal:'
print '    Development (train):', len(sigDataDev)
print '    Validation (test):', len(sigDataVal)
print '  Background:'
print '    Development (train):', len(bkgDataDev)
print '    Validation (test):', len(bkgDataVal)

print 'Finding features of interest'
trainFeatures = [var for var in data.columns if var in myFeatures]
otherFeatures = [var for var in data.columns if var not in trainFeatures]

print "Filtering the features of interest"
tmpList = list(trainFeatures) # To create a real copy
tmpList.append("category") # Add to tmpList any columns that really are needed, for whatever reason
tmpList.append("weight")
data = data[tmpList]

dataDev = sigDataDev[tmpList].copy()
dataDev = dataDev.append(bkgDataDev[tmpList].copy(), ignore_index=True)

dataVal = sigDataVal[tmpList].copy()
dataVal = dataVal.append(bkgDataVal[tmpList].copy(), ignore_index=True)

print data.describe()
print dataDev.describe()
print dataVal.describe()

######################################

print "Preparing the data for the NN"
XDev = dataDev.ix[:,0:len(trainFeatures)]
XVal = dataVal.ix[:,0:len(trainFeatures)]
YDev = np.ravel(dataDev.category)
YVal = np.ravel(dataVal.category)
weightDev = np.ravel(dataDev.weight)
weightVal = np.ravel(dataVal.weight)

print("Fitting the scaler and scaling the input variables")
scaler = StandardScaler().fit(XDev)
XDev = scaler.transform(XDev)
XVal = scaler.transform(XVal)

scalerfile = 'scaler.sav'
joblib.dump(scaler, scalerfile)


compileArgs = {'loss': 'binary_crossentropy', 'optimizer': 'adam', 'metrics': ["accuracy"]}
trainParams = {'epochs': 40, 'batch_size': 40, 'verbose': 1}

def getDefinedClassifier(nIn, nOut, compileArgs):
  model = Sequential()
  model.add(Dense(16, input_dim=nIn, kernel_initializer='he_normal', activation='relu'))
  #model.add(Dropout(0.5))
  model.add(Dense(10, kernel_initializer='he_normal', activation='relu'))
  #model.add(Dropout(0.5))
  model.add(Dense(nOut, activation="sigmoid", kernel_initializer='glorot_normal'))
  model.compile(**compileArgs)
  return model

print type(XVal)
print XVal.shape
print XVal.dtype

print("Starting the training")
start = time.time()
model = getDefinedClassifier(len(trainFeatures), 1, compileArgs)
model.fit(XDev, YDev, validation_data=(XVal,YVal,weightVal), sample_weight=weightDev, **trainParams)
print("Training took ", time.time()-start, " seconds")

name = "myNN"
model.save(name+".h5")
#model_json = model.to_json()
#with open(name + ".json", "w") as json_file:
#  json_file.write(model_json)
#model.save_weights(name + ".h5")

## To load:
#from keras.models import model_from_json
#with open('model.json', 'r') as json_file:
#  loaded_model_json = json_file.read()
#loaded_model = model_from_json(loaded_model_json)
#loaded_model.load_weights("model.h5")

print("Getting predictions")
devPredict = model.predict(XDev)
valPredict = model.predict(XVal)

print("Getting scores")

scoreDev = model.evaluate(XDev, YDev, sample_weight=weightDev, verbose = 1)
scoreVal = model.evaluate(XVal, YVal, sample_weight=weightVal, verbose = 1)
print ""

print "Dev score:", scoreDev
print "Val score:", scoreVal
print confusion_matrix(YVal, valPredict.round())
print cohen_kappa_score(YVal, valPredict.round())

import sys
sys.exit("Done!")

#########################################################

# Let's repeat the above, but monitor the evolution of the loss function
import matplotlib.pyplot as plt

history = model.fit(XDev, YDev, validation_data=(XVal,YVal,weightVal), sample_weight=weightDev, **trainParams)
print(history.history.keys())

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

sys.exit("Done!")

#########################################################

print "Attempting KFold"

from sklearn.model_selection import StratifiedKFold

seed = 42
np.random.seed(seed)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
cvscores = []
for train, test in kfold.split(XDev, YDev):
  model = getDefinedClassifier(len(trainFeatures), 1, compileArgs)
  model.fit(XDev[train], YDev[train], validation_data=(XDev[test],YDev[test],weightDev[test]), sample_weight=weightDev[train], **trainParams)
  scores = model.evaluate(XDev[test], YDev[test], sample_weight=weightDev[test], verbose=1)
  print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
  cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))



