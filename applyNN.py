from keras.models import load_model
import ROOT
import numpy as np
import pandas as pd
import root_numpy
from sklearn.externals import joblib

filesToProcess=["Wjets_200to400_test_skimmed", "T2DegStop_300_270_test_skimmed"]
myFeatures = ["Jet1Pt", "Met", "Njet", "LepPt", "LepEta", "LepChg", "HT", "NbLoose"]

model = load_model("myNN.h5")
scalerfile = 'scaler.sav'
scaler = joblib.load(scalerfile)
treeName = "bdttree"

for fileName in filesToProcess:
  Data = pd.DataFrame(root_numpy.root2array(fileName + ".root",  treename=treeName))
  trainFeatures = [var for var in Data.columns if var in myFeatures]

  X = Data[trainFeatures].ix[:,0:len(trainFeatures)]
  X = scaler.transform(X)
  Predictions = model.predict(X)
  Data["NN"] = Predictions
  print Data.head()
  Data = Data.values
  print Data.type
  tree = root_numpy.array2tree(Data, name=treeName)
  root_numpy.array2root(Data, fileName + "_nn.root", treeName, "RECREATE")
