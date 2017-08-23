from keras.optimizers import Adam
import time
import keras
import pandas
from keras.models import Sequential
from keras.layers import Dense, Dropout, AlphaDropout
from sklearn.metrics import confusion_matrix, cohen_kappa_score
#from scipy.stats import ks_2samp

from prepareDATA import * 

compileArgs = {'loss': 'binary_crossentropy', 'optimizer': 'adam', 'metrics': ["accuracy"]}
trainParams = {'epochs': 2, 'batch_size': 20, 'verbose': 1}
learning_rate = 0.001/5.0
myAdam = Adam(lr=learning_rate)
compileArgs['optimizer'] = myAdam

def getDefinedClassifier(nIn, nOut, compileArgs):
  model = Sequential()
  model.add(Dense(7, input_dim=nIn, kernel_initializer='he_normal', activation='relu'))
  #model.add(Dropout(0.2))
  #model.add(Dense(10, kernel_initializer='he_normal', activation='relu'))
  #model.add(Dropout(0.2))
  model.add(Dense(nOut, activation="sigmoid", kernel_initializer='glorot_normal'))
  model.compile(**compileArgs)
  return model

def getSELUClassifier(nIn, nOut, compileArgs):
  model = Sequential()
  model.add(Dense(16, input_dim=nIn, kernel_initializer='he_normal', activation='selu'))
  model.add(AlphaDropout(0.2))
  model.add(Dense(32, kernel_initializer='he_normal', activation='selu'))
  model.add(AlphaDropout(0.2))
  model.add(Dense(32, kernel_initializer='he_normal', activation='selu'))
  model.add(AlphaDropout(0.2))
  model.add(Dense(32, kernel_initializer='he_normal', activation='selu'))
  model.add(AlphaDropout(0.2))
  model.add(Dense(nOut, activation="sigmoid", kernel_initializer='glorot_normal'))
  model.compile(**compileArgs)
  return model

print type(XVal)
print XVal.shape
print XVal.dtype



print("Starting the training")
start = time.time()
call = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-7, patience=5, verbose=1, mode='auto')
model = getDefinedClassifier(len(trainFeatures), 1, compileArgs)
#model = getSELUClassifier(len(trainFeatures), 1, compileArgs)
history = model.fit(XDev, YDev, validation_data=(XVal,YVal,weightVal), sample_weight=weightDev,callbacks=[call], **trainParams)
print("Training took ", time.time()-start, " seconds")

# To save:
name = "myNN_DM30"
model.save(name+".h5")
model_json = model.to_json()
with open(name + ".json", "w") as json_file:
  json_file.write(model_json)
model.save_weights(name + ".h5")

# To load:
'''
from keras.models import model_from_json
with open(name + '.json', 'r') as json_file:
    loaded_model_json = json_file.read()
model = model_from_json(loaded_model_json)
model.load_weights(name+".h5")
model.compile(loss = 'binary_crossentropy', optimizer = 'adam')
'''

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

sigDataVal = dataVal[dataVal.category==1]
bkgDataVal = dataVal[dataVal.category==0]

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

fomEvo = []
fomCut = []

bkgEff = []
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

sigEff = []

sig_Init = dataVal[dataVal.category == 1].weight.sum() * 35866 * 2
bkg_Init = dataVal[dataVal.category == 0].weight.sum() * 35866 * 2

for cut in np.arange(0.0, 0.9999999, 0.001):
  sig, bkg = getYields(dataVal, cut=cut)
  if sig[0] > 0 and bkg[0] > 0:
    fom, fomUnc = FullFOM(sig, bkg)
    fomEvo.append(fom)
    fomCut.append(cut)
    
max_FOM=0

print "Maximizing FOM"
for k in fomEvo:
  if k>max_FOM:
    max_FOM=k

Eff = zip(bkgEff, sigEff)

#km_value=ks_2samp((sig_dataDev["NN"].append(bkg_dataDev["NN"])),(sig_dataVal["NN"].append(bkg_dataVal["NN"])))

#print "Layers:", y
#print "Neurons:", x
#print "Cohen Kappa score:", cohen_kappa
print "Maximized FOM:", max_FOM
print "FOM Cut:", fomCut[fomEvo.index(max_FOM)]
#print "KS test statistic:", km_value[0]
#print "KS test p-value:", km_value[1]

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
