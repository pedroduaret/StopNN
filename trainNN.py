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

loc = "/home/diogo/LIP/DATA"
treeName = "bdttree"
baseSigName = "T2DegStop_300_270"
bkgDatasets = [
                "Wjets_70to100",
                "Wjets_100to200",
                "Wjets_200to400",
                "Wjets_400to600",
                "Wjets_600to800",
                "Wjets_800to1200",
                "Wjets_1200to2500",
                "Wjets_2500toInf",
                "TTJets_DiLepton",
                "TTJets_SingleLeptonFromTbar",
                "TTJets_SingleLeptonFromT",
                "ZJetsToNuNu_HT100to200",
                "ZJetsToNuNu_HT200to400",
                "ZJetsToNuNu_HT400to600",
                "ZJetsToNuNu_HT600to800",
                "ZJetsToNuNu_HT800to1200",
                "ZJetsToNuNu_HT1200to2500",
                "ZJetsToNuNu_HT2500toInf"

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
sigDataDev["sampleWeight"] = 1
sigDataVal["sampleWeight"] = 1
bkgDataDev["sampleWeight"] = 1
bkgDataVal["sampleWeight"] = 1

# Calculate event weights
# The input files already have a branch called weight, which contains the per-event weights
# These precomputed weights have all scale factors applied. If we desire to not use the scale factors
# we should compute a new set of weights ourselves. Remember to repeat for all datasets.
################### Add computation here if wanted #######################################################
# After computing the weights, the total class has to be normalized.
sigDataDev.sampleWeight = sigDataDev.weight/sigDataDev.weight.sum()
sigDataVal.sampleWeight = sigDataVal.weight/sigDataVal.weight.sum()
bkgDataDev.sampleWeight = bkgDataDev.weight/bkgDataDev.weight.sum()
bkgDataVal.sampleWeight = bkgDataVal.weight/bkgDataVal.weight.sum()


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
tmpList.append("sampleWeight")
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
weightDev = np.ravel(dataDev.sampleWeight)
weightVal = np.ravel(dataVal.sampleWeight)

print("Fitting the scaler and scaling the input variables")
scaler = StandardScaler().fit(XDev)
XDev = scaler.transform(XDev)
XVal = scaler.transform(XVal)

scalerfile = 'scaler.sav'
joblib.dump(scaler, scalerfile)


compileArgs = {'loss': 'binary_crossentropy', 'optimizer': 'adam', 'metrics': ["accuracy"]}
trainParams = {'epochs': 1, 'batch_size': 40, 'verbose': 1}

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
history = model.fit(XDev, YDev, validation_data=(XVal,YVal,weightVal), sample_weight=weightDev, **trainParams)
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


print "Calculating FOM:"
dataVal["NN"] = valPredict

def getYields(dataVal, cut=0.5, luminosity=35866, splitFactor=2):
  selectedValIdx = (dataVal.NN>cut)
  selectedVal = dataVal[selectedValIdx]

  selectedSigIdx = (selectedVal.category == 1)
  selectedBkgIdx = (selectedVal.category == 0)
  selectedSig = selectedVal[selectedSigIdx]
  selectedBkg = selectedVal[selectedBkgIdx]

  sigYield = selectedSig.weight.sum()
  sigYieldUnc = np.sqrt(np.sum(np.square(selectedSig.weight)))
  bkgYield = selectedBkg.weight.sum()
  bkgYieldUnc = np.sqrt(np.sum(np.square(selectedBkg.weight)))

  sigYield = sigYield * luminosity * splitFactor # The factor 2 comes from the splitting
  sigYieldUnc = sigYieldUnc * luminosity * splitFactor
  bkgYield = bkgYield * luminosity * splitFactor
  bkgYieldUnc = bkgYieldUnc * luminosity * splitFactor

  return ((sigYield, sigYieldUnc), (bkgYield, bkgYieldUnc))

tmpSig, tmpBkg = getYields(dataVal)
sigYield, sigYieldUnc = tmpSig
bkgYield, bkgYieldUnc = tmpBkg

print "Signal@Presel:", sigDataVal.weight.sum() * 35866 * 2
print "Background@Presel:", bkgDataVal.weight.sum() * 35866 * 2
print "Signal:", sigYield, "+-", sigYieldUnc
print "Background:", bkgYield, "+-", bkgYieldUnc

def FOM1(sIn, bIn):
  s, sErr = sIn
  b, bErr = bIn
  fom = s / (b**0.5)
  fomErr = ((sErr / (b**0.5))**2+(bErr*s / (2*(b)**(1.5)) )**2)**0.5
  return (fom, fomErr)

def FOM2(sIn, bIn):
  s, sErr = sIn
  b, bErr = bIn
  fom = s / ((s+b)**0.5)
  fomErr = ((sErr*(2*b + s)/(2*(b + s)**1.5))**2  +  (bErr * s / (2*(b + s)**1.5))**2)**0.5
  return (fom, fomErr)

def FullFOM(sIn, bIn, fValue=0.2):
  from math import log
  s, sErr = sIn
  b, bErr = bIn
  fomErr = 0.0 # Add the computation of the uncertainty later
  fomA = 2*(s+b)*log(((s+b)*(b + (fValue*b)**2))/(b**2 + (s + b) * (fValue*b)**2))
  fomB = log(1 + (s*b*b*fValue*fValue)/(b*(b+(fValue*b)**2)))/(fValue**2)
  fom = (fomA - fomB)**0.5
  return (fom, fomErr)

print "Basic FOM (s/SQRT(b)):", FOM1((sigYield, sigYieldUnc), (bkgYield, bkgYieldUnc))
print "Basic FOM (s/SQRT(s+b)):", FOM2((sigYield, sigYieldUnc), (bkgYield, bkgYieldUnc))
print "Full FOM:", FullFOM((sigYield, sigYieldUnc), (bkgYield, bkgYieldUnc))

import sys
#sys.exit("Done!")

#########################################################

# Let's repeat the above, but monitor the evolution of the loss function
import matplotlib.pyplot as plt

#history = model.fit(XDev, YDev, validation_data=(XVal,YVal,weightVal), sample_weight=weightDev, **trainParams)
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

fomEvo = []
fomCut = []
for cut in np.arange(0.0, 0.9999999, 0.001):
  sig, bkg = getYields(dataVal, cut=cut)
  if sig[0] > 0 and bkg[0] > 0:
    fom, fomUnc = FullFOM(sig, bkg)
    fomEvo.append(fom)
    fomCut.append(cut)

plt.plot(fomCut, fomEvo)
plt.title("FOM")
plt.ylabel("FOM")
plt.xlabel("ND")
plt.legend(['test'], loc='upper left')
plt.show()

#print fomEvo

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



