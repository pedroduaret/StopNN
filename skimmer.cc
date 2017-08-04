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

  TTree* tmp = static_cast<TTree*>(inFile.Get("bdttree"));
  tmp->SetBranchStatus("*",1);
  tmp->SetBranchStatus("*_Up",0);
  tmp->SetBranchStatus("*_Down",0);

  TTree* newTree = tmp->CloneTree(75000);
  //TTree* newTree = tmp->CloneTree();

  newTree->Write();

  return 0;
}
