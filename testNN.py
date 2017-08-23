import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import root_numpy
import numpy as np
import pandas
import keras
import time
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, AlphaDropout
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import localConfig as cfg
from commonFunctions import StopDataLoader, FOM1, FOM2, FullFOM, getYields
import sys

from keras.models import model_from_json
from prepareDATA import *

model_name = "myNN_N13_L2_E25_DevDM30_Val550_520"

print "Loading Model ..."
with open(model_name+'.json', 'r') as json_file:
  loaded_model_json = json_file.read()
model = model_from_json(loaded_model_json)
model.load_weights(model_name+".h5")
model.compile(loss = 'binary_crossentropy', optimizer = 'adam')

print("Getting predictions")
devPredict = model.predict(XDev)
valPredict = model.predict(XVal)


print("Getting scores")

scoreDev = model.evaluate(XDev, YDev, sample_weight=weightDev, verbose = 1)
scoreVal = model.evaluate(XVal, YVal, sample_weight=weightVal, verbose = 1)
print ""
cohen_kappa=cohen_kappa_score(YVal, valPredict.round())
   
print "Calculating parameters"
dataDev["NN"] = devPredict
dataVal["NN"] = valPredict  

sig_dataDev=dataDev[dataDev.category==1]
bkg_dataDev=dataDev[dataDev.category==0]
sig_dataVal=dataVal[dataVal.category==1]
bkg_dataVal=dataVal[dataVal.category==0]

tmpSig, tmpBkg = getYields(dataVal)
sigYield, sigYieldUnc = tmpSig
bkgYield, bkgYieldUnc = tmpBkg
        

fomEvo = []
fomCut = []

bkgEff = []
sigEff = []

sig_Init = dataVal[dataVal.category == 1].weight.sum() * 35866 * 2
bkg_Init = dataVal[dataVal.category == 0].weight.sum() * 35866 * 2

for cut in np.arange(0.0, 0.9999, 0.001):
    sig, bkg = getYields(dataVal, cut=cut, luminosity=luminosity)
    if sig[0] > 0 and bkg[0] > 0:
        fom, fomUnc = FullFOM(sig, bkg)
        fomEvo.append(fom)
        fomCut.append(cut)
        bkgEff.append(bkg[0]/bkg_Init)
        sigEff.append(sig[0]/sig_Init)

max_FOM=0

for k in fomEvo:
    if k>max_FOM:
        max_FOM=k

Eff = zip(bkgEff, sigEff)

km_value=ks_2samp((sig_dataDev["NN"].append(bkg_dataDev["NN"])),(sig_dataVal["NN"].append(bkg_dataVal["NN"])))
        
#f.write(str(y)+"\n")           
#print "Layers:", y
#f.write(str(x)+"\n")
#print "Neurons:", x
#f.write(str(cohen_kappa)+"\n")
print "Cohen Kappa score:", cohen_kappa
#f.write(str(max_FOM)+"\n")
print "Maximized FOM:", max_FOM
#f.write(str(fomCut[fomEvo.index(max_FOM)])+"\n")
print "FOM Cut:", fomCut[fomEvo.index(max_FOM)]
#f.write(str(km_value[0])+"\n")
print "KS test statistic:", km_value[0]
#f.write(str(km_value[1])+"\n")
print "KS test p-value:", km_value[1]
        
        #f.close()
        
selectedVal = dataVal[dataVal.NN>fomCut[fomEvo.index(max_FOM)]]
selectedSig = selectedVal[selectedVal.category == 1]
selectedBkg = selectedVal[selectedVal.category == 0]
sigYield = selectedSig.weight.sum()
bkgYield = selectedBkg.weight.sum()
sigYield = sigYield * luminosity * 2          #The factor 2 comes from the splitting
bkgYield = bkgYield * luminosity * 2
        

print "Selected events left after cut @", fomCut[fomEvo.index(max_FOM)]
print "   Number of selected Signal Events:", len(selectedSig)
print "   Number of selected Background Events:", len(selectedBkg)
print "   Sig Yield", sigYield
print "   Bkg Yield", bkgYield
        
print "Plotting"

plt.figure(figsize=(7,6))
plt.hist(sig_dataDev["NN"], 50, facecolor='blue', alpha=0.7, normed=1, weights=sig_dataDev["weight"])
plt.hist(bkg_dataDev["NN"], 50, facecolor='red', alpha=0.7, normed=1, weights=bkg_dataDev["weight"])
plt.hist(sig_dataVal["NN"], 50, color='blue', alpha=1, normed=1, weights=sig_dataVal["weight"], histtype="step")
plt.hist(bkg_dataVal["NN"], 50, color='red', alpha=1, normed=1, weights=bkg_dataVal["weight"], histtype="step")
plt.xlabel('NN output')
plt.suptitle("MVA overtraining check for classifier: NN", fontsize=13, fontweight='bold')
plt.title("Cohen's kappa: {0}\nKolmogorov Smirnov test: {1}".format(cohen_kappa, km_value[1]), fontsize=10)
plt.legend(['Signal (Test sample)', 'Background (Test sample)', 'Signal (Train sample)', 'Background (Train sample)\nasdfgh'], loc='upper right')
plt.savefig('hist_'+model_name+'.png', bbox_inches='tight')
plt.show()


both_dataDev=bkg_dataDev.append(sig_dataDev)
plt.figure(figsize=(7,6))
plt.xlabel('NN output')
plt.title("Number of Events")
#plt.yscale('log', nonposy='clip')
plt.legend(['Background + Signal (test sample)', 'Background (test sample)'], loc="upper left" )
plt.hist(bkg_dataDev["NN"], 50, facecolor='red', weights=bkg_dataDev["weight"])
plt.hist(both_dataDev["NN"], 50, color="blue", histtype="step", weights=both_dataDev["weight"])
plt.savefig('pred_'+model_name+'.png', bbox_inches='tight')
plt.show()
        
plt.figure(figsize=(7,6))
plt.subplots_adjust(hspace=0.5)
plt.subplot(211)
plt.plot(fomCut, fomEvo)
plt.title("FOM")
plt.ylabel("FOM")
plt.xlabel("ND")
plt.legend(["Max. FOM: {0}".format(max_FOM)], loc='upper left')


plt.subplot(212)
plt.semilogy(fomCut, Eff)
plt.axvspan(fomCut[fomEvo.index(max_FOM)], 1, facecolor='#2ca02c', alpha=0.3)
#plt.axvline(x=fomCut[fomEvo.index(max_FOM)], ymin=0, ymax=1)
plt.title("Efficiency")
plt.ylabel("Eff")
plt.xlabel("ND")
plt.legend(['Background', 'Signal'], loc='upper left')
plt.savefig('FOM_'+model_name+'.png', bbox_inches='tight')
plt.show()
         
sys.exit("Done!")
