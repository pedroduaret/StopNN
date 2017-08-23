#!/usr/bin/env python

import root_numpy
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

import localConfig as cfg
from commonFunctions import StopDataLoader

myFeatures = ["Jet1Pt", "Met", "mt", "LepPt", "LepEta", "LepChg", "HT", "NbLoose","Njet", "JetHBpt", "DrJetHBLep", "JetHBCSV"]
inputBranches = list(myFeatures)
inputBranches.append("XS")
inputBranches.append("weight")
preselection = "(DPhiJet1Jet2 < 2.5 || Jet2Pt < 60) && (Met > 280) && (HT > 200) && (isTight == 1) && (Jet1Pt > 110)"
suffix = "_skimmed"
luminosity = 35866
number_of_events_print = 0
test_point = "550_520"
train_DM = "DM30"

print "Loading datasets..."
dataDev, dataVal = StopDataLoader(cfg.loc, inputBranches, selection=preselection, suffix=suffix, signal=train_DM, test=test_point)
#print dataDev.describe()
#print dataVal.describe()

if number_of_events_print == 1:
    print 'Datasets contain a total of', len(data), '(', data.weight.sum()*luminosity, 'weighted) events:'
    print '  Development (train):', len(dataDev), '(', dataDev.weight.sum()*luminosity, 'weighted)'
    print '    Signal:', len(dataDev[dataDev.category == 1]), '(', dataDev[dataDev.category == 1].weight.sum()*luminosity, 'weighted)'
    print '    Background:', len(dataDev[dataDev.category == 0]), '(', dataDev[dataDev.category == 0].weight.sum()*luminosity, 'weighted)'
    print '  Validation (test):', len(dataVal), '(', dataVal.weight.sum()*luminosity, 'weighted)'
    print '    Signal:', len(dataVal[dataVal.category == 1]), '(', dataVal[dataVal.category == 1].weight.sum()*luminosity, 'weighted)'
    print '    Background:', len(dataVal[dataVal.category == 0]), '(', dataVal[dataVal.category == 0].weight.sum()*luminosity, 'weighted)'

data = dataDev.copy()
data = data.append(dataVal.copy(), ignore_index=True)

print 'Finding features of interest'
trainFeatures = [var for var in data.columns if var in myFeatures]
otherFeatures = [var for var in data.columns if var not in trainFeatures]

print "Preparing the data for the NN"
XDev = dataDev.ix[:,0:len(trainFeatures)]
XVal = dataVal.ix[:,0:len(trainFeatures)]
YDev = np.ravel(dataDev.category)
YVal = np.ravel(dataVal.category)
weightDev = np.ravel(dataDev.sampleWeight)
weightVal = np.ravel(dataVal.sampleWeight)

print "Fitting the scaler and scaling the input variables"
scaler = StandardScaler().fit(XDev)
XDev = scaler.transform(XDev)
XVal = scaler.transform(XVal)

scalerfile = 'scaler_'+train_DM+'.sav'
joblib.dump(scaler, scalerfile)

print "DATA is ready!"
