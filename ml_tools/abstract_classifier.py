import os
from abc import ABC
import pandas as pd


from ROOT import kBlue, kRed, kBlack
from ROOT import TCanvas, TH1F, TFile
from sklearn.model_selection import train_test_split

import ml_tools.bc_tools as bc_tools

def remove_suffix(text, suffix):
    if text.endswith(suffix):
        return text[:len(text)-len(suffix)]
    return text  # or whatever

class Abstract_Classifier(ABC):
    def __init__(self,*args,**kwargs):
        self.features=kwargs.get("features")
        self.channels=kwargs.get("channels")
        self.path=kwargs.get("csv_files_path")
        self.bkg_names=kwargs.get("bkg_names")
        self.signal_names=kwargs.get("signal_names")
        self._prepare_data()
    
    def _prepare_data(self):
        self.signal_data=bc_tools.concat_channels(
            self.path,
            self.signal_names,
            self.channels,
            self.features
        )
        
        self.bkg_data=bc_tools.concat_channels(
            self.path,
            self.bkg_names,
            self.channels,
            self.features
        )
        
        pred1 , labels1= bc_tools.prepare_to_train(self.signal_data,self.bkg_data)
        trainPred, testPred, trainLab, testLab = train_test_split(pred1, labels1, test_size=0.30)
        
        self.testLab=testLab
        self.trainLab=trainLab
        self.testPred=testPred
        self.trainPred=trainPred
    
    def get_features_corr(self):
        feats=[]
        for feat in self.features.copy():
            if (feat == "light_jets_multiplicity"):
                feats+=["$n_j$"]
            else:
                feats+=["$"+feat.replace("#","\\")+"$"]
            
        df = pd.DataFrame(
            self.bkg_data, 
            columns = feats
        )
        return df.corr()
        
    def _draw_discrtiminator(self,name,path_to_save):
        c1 = TCanvas( f'c-{name}', '', 0, 0, 1280, 720)
        c1.SetGrid()
        c1.SetLogy()
        colors = [kBlue,kRed,3, 7, 6, kBlack, 2,  9, 1, 43, 97, 38, 3, 7, 6, kBlack, 2, 4, 8]
        hist_dict={}
        scores_dict={}
        for i, channel in enumerate(self.channels):
            h = TH1F(
                f"{name}_{channel}",
                f"{name};ML-score;nevents(137/fb)", 
                100, 0.0,1.0
            )
            h.SetLineWidth(1)
            h.SetDirectory(0)
            h.SetLineColor(kBlack)
            h.SetFillColor(colors[i])
            data_to_eval=bc_tools.concat_channels(
                self.path,
                [name],
                self.channels,
                self.features
            )
            
            scores=self.model.predict_proba(data_to_eval)
            df=pd.DataFrame(scores, columns = ['scores'])
            scores_dict[channel]=df
            
            for score in scores:
                h.Fill(score)
            try: 
                h.Scale(
                    bc_tools.get_yield(self.path,[name],channel)/h.Integral()
                )
            except:
                pass
            hist_dict[channel]=h
        h.Draw("hist")
        df.to_csv(os.path.join(path_to_save,f"{self.model_name}_{name}.csv"),index=False)
        c1.SaveAs(os.path.join(path_to_save,f"{self.model_name}_{name}.png"))
        return (name,hist_dict)
    
    def get_discriminator_histograms(self,path_to_save):
        def mapping(name):
            return self._draw_discrtiminator(name,path_to_save)
        self.histograms = dict(map(mapping , self.signal_names + self.bkg_names))
        
        f = TFile(os.path.join(path_to_save,f"{self.model_name}.root"),"RECREATE")
        for signal in self.histograms:
            h=self.histograms[signal][self.channels[0]]
            h.Write(remove_suffix(h.GetName(),f"_{self.channels[0]}"))
        self.model.save_model(os.path.join(path_to_save,f"{self.model_name}.joblib"))
        return self.histograms
        
    def fit_model(self):
        self.model.fit(self.trainPred, self.trainLab)