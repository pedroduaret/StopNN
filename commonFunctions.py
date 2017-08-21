import root_numpy
import pandas

signalMap = {
              "DM30" : [],
              "300_270" : ["T2DegStop_300_270"],
            }
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


def StopDataLoader(path, features, selection="", treename="bdttree", suffix="", signal="DM30", fraction=1.0):
  if signal not in signalMap:
    raise KeyError("Unknown signal requested ("+signal+")")
  if fraction >= 1.0:
    fraction = 1.0
  if fraction < 0.0:
    raise ValueError("An invalid fraction was chosen")
  if "XS" not in features:
    features.append("XS")
  if "weight" not in features:
    features.append("weight")



  sigDev = None
  sigVal = None
  for sigName in signalMap[signal]:
    stopM = int(sigName[10:13])
    tmp = root_numpy.root2array(
                                path + "/train/" + sigName + suffix + ".root",
                                treename=treename,
                                selection=selection,
                                branches=features
                                )
    if fraction < 1.0:
      tmp = tmp[:int(len(tmp)*fraction)]
    if sigDev is None:
      sigDev = pandas.DataFrame(tmp)
      sigDev["stopM"] = stopM
    else:
      tmp2 = pandas.DataFrame(tmp)
      tmp2["stopM"] = stopM
      sigDev = sigDev.append(tmp2, ignore_index=True)

    tmp = root_numpy.root2array(
                                path + "/test/" + sigName + suffix + ".root",
                                treename=treename,
                                selection=selection,
                                branches=features
                                )
    if fraction < 1.0:
      tmp = tmp[:int(len(tmp)*fraction)]
    if sigVal is None:
      sigVal = pandas.DataFrame(tmp)
      sigVal["stopM"] = stopM
    else:
      tmp2 = pandas.DataFrame(tmp)
      tmp2["stopM"] = stopM
      sigVal = sigVal.append(tmp2, ignore_index=True)

  bkgDev = None
  bkgVal = None
  for bkgName in bkgDatasets:
    tmp = root_numpy.root2array(
                                path + "/train/" + bkgName + suffix + ".root",
                                treename=treename,
                                selection=selection,
                                branches=features
                                )
    if fraction < 1.0:
      tmp = tmp[:int(len(tmp)*fraction)]
    if bkgDev is None:
      bkgDev = pandas.DataFrame(tmp)
    else:
      bkgDev = bkgDev.append(pandas.DataFrame(tmp), ignore_index=True)

    tmp = root_numpy.root2array(
                                path + "/test/" + bkgName + suffix + ".root",
                                treename=treename,
                                selection=selection,
                                branches=features
                                )
    if fraction < 1.0:
      tmp = tmp[:int(len(tmp)*fraction)]
    if bkgVal is None:
      bkgVal = pandas.DataFrame(tmp)
    else:
      bkgVal = bkgVal.append(pandas.DataFrame(tmp), ignore_index=True)

  sigDev["category"] = 1
  sigVal["category"] = 1
  bkgDev["category"] = 0
  bkgVal["category"] = 0
  sigDev["sampleWeight"] = 1
  sigVal["sampleWeight"] = 1
  bkgDev["sampleWeight"] = 1
  bkgVal["sampleWeight"] = 1

  bkgDev["stopM"] = -1
  bkgVal["stopM"] = -1

  if fraction < 1.0:
    sigDev.weight = sigDev.weight/fraction
    sigVal.weight = sigVal.weight/fraction
    bkgDev.weight = bkgDev.weight/fraction
    bkgVal.weight = bkgVal.weight/fraction

  sigDev.sampleWeight = sigDev.weight/sigDev.XS
  sigVal.sampleWeight = sigVal.weight/sigVal.XS
  bkgDev.sampleWeight = bkgDev.weight
  bkgVal.sampleWeight = bkgVal.weight

  sigDev.sampleWeight = sigDev.sampleWeight/sigDev.sampleWeight.sum()
  sigVal.sampleWeight = sigVal.sampleWeight/sigVal.sampleWeight.sum()
  bkgDev.sampleWeight = bkgDev.sampleWeight/bkgDev.sampleWeight.sum()
  bkgVal.sampleWeight = bkgVal.sampleWeight/bkgVal.sampleWeight.sum()

  dev = sigDev.copy()
  dev = dev.append(bkgDev.copy(), ignore_index=True)
  val = sigVal.copy()
  val = val.append(bkgVal.copy(), ignore_index=True)

  return dev, val
