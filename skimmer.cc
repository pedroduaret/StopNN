#include "TROOT.h"
#include "TTree.h"
#include "TFile.h"

#include <string>
#include <iostream>

//int main(int argc, char** argv)
int skimmer(std::string fileToProcess)
{
  //std::string fileToProcess = "";

  TFile inFile((fileToProcess + ".root").c_str(), "READ");
  TFile outFile((fileToProcess + "_skimmed.root").c_str(), "RECREATE");

  TTree* inTree = static_cast<TTree*>(inFile.Get("bdttree"));
  inTree->SetBranchStatus("*",1);
  inTree->SetBranchStatus("*_Up",0);
  inTree->SetBranchStatus("*_Down",0);

  TTree* tmp = inTree->CloneTree(75000);
  tmp->SetBranchStatus("*", 1);
  tmp->SetBranchStatus("weight", 0);
  TTree* outTree = tmp->CloneTree();
  tmp->SetBranchStatus("weight", 1);

  Float_t weight = 0;
  tmp->SetBranchAddress("weight", &weight);
  TBranch* newWeightBranch = outTree->Branch("weight", &weight);

  Long64_t inEvents = inTree->GetEntries();
  Long64_t copyEvents = tmp->GetEntries();
  Float_t factor = 1;
  if(copyEvents < inEvents)
  {
    factor = inEvents;
    factor = factor / copyEvents;
  }
  std::cout << "The scaling factor is: " << factor << std::endl;

  for(Long64_t i = 0; i < copyEvents; ++i)
  {
    tmp->GetEntry(i);
    weight = weight * factor;
    newWeightBranch->Fill();
  }


  outTree->Write();

  return 0;
}
