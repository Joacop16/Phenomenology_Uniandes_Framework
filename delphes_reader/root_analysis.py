import os
import ROOT
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from ROOT import TCanvas #It is necessary to show plots with ROOT.
from ROOT import TH1F #It is necessary to plot histograms with ROOT.
from ROOT import THStack #It is necessary to plot many histograms at the same time with ROOT.
from ROOT import TLegend #It is necessary to plot labels when you plot many histograms at the same time with ROOT.
from ROOT import TFile #It is necessary to save histograms in a .root file.

import uproot

def get_kinematics_row(particle_list:list):
    ''' Extracts main kinematic variables of a particle (or more) and returns a dictionary with them.
    Parameters:
        *args (Particle): arguments could be many particle as it want to get its kinematics variable. For example: muons[0],muons[1] where muons is a list of muon particles.
    Return:
        Python dictionary: Contains main kinematic variables.
    '''  
    particles= list(particle_list)
    co_particles=particles.copy()
    row = {} #{'header': value}
    for particle in particles:
        name=particle.Name
        row[f"pT_{{{name}}}(GeV)"]=particle.pt()
        row[f"#eta_{{{name}}}"]=particle.eta()
        row[f"#phi_{{{name}}}"]=particle.phi()
        row[f"Energy_{{{name}}}(GeV)"]=particle.energy()
        row[f"Mass_{{{name}}}(GeV)"]=particle.m()
        co_particles.pop(0)
        for co_particle in co_particles:
            co_name=co_particle.Name
            row[f"#Delta R_{{{name}{co_name}}}"]=particle.DeltaR(co_particle)
            row[f"#Delta #eta_{{{name}{co_name}}}"]=particle.DeltaEta(co_particle)
            row[f"#Delta #phi_{{{name}{co_name}}}"]=particle.DeltaPhi(co_particle)
            row[f"#Delta pT_{{{name}{co_name}}}(GeV)"]=particle.sDeltaPT(co_particle)
            row[f"#Delta #vec{{pT}}_{{{name}{co_name}}}(GeV)"]=particle.vDeltaPT(co_particle)
            row[f"#Delta #vec{{p}}_{{{name}{co_name}}}(GeV)"]=particle.vDeltaP(co_particle)
    return row

default_hist_bins_dict={
    "#Delta R":[96,0,7],
    "#Delta #eta":[80,-5,5],
    "#Delta #phi":[52,-3.25,3.25],
    "#Delta pT":[120, 0.0, 1500.0],
    "#Delta #vec{pT}":[240, 0.0, 4800.0],
    "#Delta #vec{p}":[240, 0.0, 4800.0],
    "MET(GeV)":[80, 0.0, 1000.0],
    "pT_": [160, 0.0, 2000.0],
    "sT(GeV)": [200, 0.0, 4000.0],
    "mT(GeV)": [200, 0.0, 4000.0],
    "#eta_":[80, -5, 5],
    "#phi_":[128, -3.2, 3.2],
    "Energy_":[80, 0.0, 1000.0]
}

def make_histograms(df,integral=1.0,hist_bins_dict=default_hist_bins_dict):
    ''' Uses ROOT to create histograms using all data contained in a DataFrame.  
    Parameters:
        df (DataFrame): It is a DataFrame where each row correspond to a different particle and each column to its corresponding kinematic variable value.
    Return:
        TH1F (Python dictionary): Contains all histograms. The keys of this dictionary are kinematic variable names (the same keys of the DataFrame).
    '''  
    hist_dict={}
    for key in dict(df).keys():
        for hist_key in hist_bins_dict.keys():
            if hist_key in key:
                bins=hist_bins_dict.get(hist_key,False)
                break
        if not (bins): 
            print("No Histogram Dictionary for ",key)
        else: 
            x_axis = key.replace('(' , "[" ).replace(')' , "]" )
            h=ROOT.TH1F(
                key, f"; {x_axis}; Events",
                bins[0],bins[1],bins[2]
            )
            h.SetDirectory(0)
            for value in df[key]:
                h.Fill(value)
            try: 
                h.Scale(integral/h.Integral())
            except:
                print(h.GetName(), " is empty!")
            hist_dict.update({key : h})
    return hist_dict

def histos_matplotlib(Dataset, column_key, log = False, c = 'blue', file_name = '', nbins = 100):
    ''' Uses matplotlib to create histograms using all data contained in a column of a DataFrame.  
    Parameters:
        Dataset (DataFrame)*: It is a DataFrame where each row correspond to a different particle and each column to its corresponding kinematic variable value.
        column_key (string)*: It is the key of the column that we want to plot as a histogram.
        log (Boolean): If it is True, the Dataset will be readed using log 10 scale.
        c (string): Histogram color.
        file_name (string): File name that would be used to save the plot.
        nbins (float): Bins number.   
    '''  
    Data = Dataset[column_key]
    if log: Data = np.log10(Data)
    
    fig = plt.figure(figsize = (6,4))
    plt.hist(Data, bins = nbins, color = c, density=True)
    
    name = '$' + column_key.replace('#' , "\\" ).replace('(' , "[" ).replace(')' , "]" ) + '$'
    plt.xlabel(fr'{name}', fontsize = 12)
    plt.ylabel('A.U', fontsize = 12)
    
    Statistics = 'Mean = ' + str(round(Data.mean(),3))+ ", STD = " + str(round(Data.std(),3))
    plt.title(Statistics, loc = 'right', fontsize = 12)
    
    if file_name != '': plt.savefig(file_name, bbox_inches='tight')
    
    plt.show()
    
def overlap_histos(kinematic_variable, Dict_Histos, alpha = 0.05, Stack = False, Log = False, Grid = False):
    ''' Uses ROOT to overlap histograms using all kinematic variable's histograms contained in a Dicctionary.
    Parameters:
        kinematic_variable (string)*: Name of the kinematic variable. It must be also the key to access the corresponding histograms inside Dict_Histos.
        Dict_Histos (Python Dicctionary)*: It is the dicctionary that contains all the histograms. 
        This Dicctionary should have keys with the name of the signals, and each signal should have other dictionaries with the same structure as an output of make_histograms.
        alpha (float): Histogram transparency. It must be between 0 and 1.
        Stack (Boolean): If it is True, the plot of histograms will consider a Stack between them.
        Log (Boolean): If it is True, the histogram will be plotted using log 10 scale.   
        Grid (Boolean): If it is True, the canvas will plot a grid in the graphic.
    Return:
        THStack: Histos contains all the histograms with overlap or stack.
        THCanvas: canvas allows to edit some features such as x limit after using this function.
        THLegend: legend contains the labels of each signal.
    '''        
    canvas = TCanvas('','', 600, 400)
    legend = TLegend(0.6,.8,0.89,.89)
    legend.SetNColumns(4)
    legend.SetLineWidth(1)

    Histos = THStack('hist', '')
    
    for i in range(len(Dict_Histos.keys())):
        signal = list(Dict_Histos.keys())[i]
                
        if (Dict_Histos[signal] != {}):
            histo = Dict_Histos[signal][kinematic_variable]
            histo.SetLineColor(i+1)
            histo.SetFillColorAlpha(i+1, alpha)
            histo.SetLineWidth(2)
            histo.SetDirectory(0)
            Histos.Add(histo)
            legend.AddEntry(histo,signal)
    
    x_axis = kinematic_variable.replace('(' , "[" ).replace(')' , "]" )
    
    if (Stack):
        Histos.Draw("hist")
        Histos.SetTitle(f'; {x_axis}; Events')
    else:
        Histos.Draw("histnostack")
        Histos.SetTitle(f'; {x_axis}; A.U')
    
    if Log: 
        canvas.SetLogy()
        Histos.SetMinimum(10)
        Histos.SetMaximum(1e8)
        
    if Grid: canvas.SetGrid()
    
    canvas.Draw()
    legend.Draw('same')
    
    return Histos, canvas, legend

def sum_histos(histo_list):
    ''' Uses ROOT to sum histograms using all histograms contained in a list. Each histogram must be normalized by the number of physical events.
    Parameters:
        histo_list (TH1F list)*: Contains all histograms that will be summed.
    Return:
        TH1F: Histogram that is the sum of all histograms in the list.
    '''            
    result = TH1F('sum', 'sum', histo_list[0].GetNbinsX(),0.0,histo_list[0].GetBinWidth(0)*histo_list[0].GetNbinsX())
    result.SetDirectory(0)
    
    for histo in histo_list:
        for i in range(histo.GetNbinsX()):
            #Sumemos los bines
            sum_ = result.GetBinContent(i + 1)
            sum_ = sum_ + histo.GetBinContent(i + 1)
            result.SetBinContent(i+1, sum_)
            #Sumemos el error
            err_ = result.GetBinError(i + 1)
            err_ = err_ + histo.GetBinError(i+1)
            result.SetBinError(i+1, err_)
        
    return result

def generate_csv(dictionary_list,file_name):
    ''' Uses Pandas to create a csv file using all data contained in a list of dictionaries.  
    Parameters:
        dictionary_list (Python list): It is a list where each member is a dictionary with the structure of get_kinematics_row outputs.
        file_name (string): It is the name that the .csv file will have.
    '''      
    Data = pd.DataFrame()
    for dictionary_kinematics in dictionary_list:
        row = pd.DataFrame.from_dict(dictionary_kinematics, orient = "index").T
        Data = pd.concat([Data,row]) 
        Data.reset_index(drop=True, inplace=True)
    Data.to_csv(file_name, index= False)
    
def Save_Histograms_png(path_to_save, Dict_Hist, Log_Y = False):
    ''' Uses Root to save all histograms contained in a python dictionary (Dict_Hist) as .png files. This function uses the keys of the dictionary to save each histogram.  
    Parameters:
        path_to_save (string): Folder name that will be used to save all histogramas as .png files.
        Dict_Hist (Python Dictionary)*: It is the dictionary that contains all the histograms. 
        Log_Y (Boolean): If it is True, the histogram will be plotted using log 10 Y-scale.   
    '''  
    for key in Dict_Hist.keys():
        histo = Dict_Hist[key]
        canvas = TCanvas(key, " ", 0, 0, 1280, 720)
        canvas.SetGrid()
        if Log_Y: canvas.SetLogy()
        histo.Draw("hist")
        canvas.SaveAs(os.path.join(path_to_save,f"histograms_{key}.png").replace('#', '').replace('{', '').replace('}', '').replace(' ', '_'))        
        
def Write_ROOT_File(path_root_file, Dict_Hist):
    ''' Uses Root to save all histograms contained in a python dictionary (Dict_Hist) in a .root file.  
    Parameters:
        path_root_file (string): .root file name that would be used to save it.
        Dict_Hist (Python Dictionary)*: It is the dictionary that contains all the histograms. 
    '''  
    ROOT_File = TFile.Open(path_root_file, 'RECREATE') 
    
    for key in Dict_Hist.keys():

        Dict_Hist[key].SetName(key)
        Dict_Hist[key].Write()

    ROOT_File.Close()
    
def Read_ROOT_File(path_root_file, expected_keys):
    ''' Uses Root to read all histograms contained in a .root file.  
    Parameters:
        path_root_file (string): .root file name that would be used to read it.
        expected_keys (Python list)*: It contains the keys that was used to save the histograms in the .root file. 
    Return:
        Python Dictionary: Dictionary that contains all the histograms.
    '''  
    Dict_hist = {}
    File = TFile.TFile.Open(path_root_file, 'READ')
    for key in expected_keys:
        histogram = File.Get(key)
        try: histogram.SetDirectory(0)
        except: pass
        Dict_hist[key] = histogram
    File.Close()
    return Dict_hist
    
def Write_txt_file_with_high_per_bin(name, Dict_Hist):
    ''' Write .dat files with the high per bin of all histograms contained in a python dictionary (Dict_Hist).
    Parameters:
        name (string): File name that will be used as preffix to save the high per bin of all histograms as .dat files.
        Dict_Hist (Python Dictionary)*: It is the dictionary that contains all the histograms. 
    '''  
    for key in Dict_Hist.keys():
        histo = Dict_Hist[key]
        
        high_list = []
        for i in range(1, histo.GetNbinsX()+1): high_list.append(histo.GetBinContent(i))  
        txt_name = f'{name}_{key}.dat'
        np.savetxt(txt_name.replace('#', '').replace('{', '').replace('}', '').replace(' ', '_'), high_list)  
        
def review_holes_in_histograms(Dict_Hist):
    ''' Returns a list with the names of all histograms with holes contained in a python dictionary (Dict_Hist). 
    Parameters:
        Dict_Hist (Python Dictionary)*: It is the dictionary that contains all the histograms. 
    Return:
        Python list: List with the names of all histograms with holes.
    '''  
    keys_histos_with_holes = []
    for key in Dict_Hist.keys():
        histo = Dict_Hist[key]
        
        for i in range(1,histo.GetNbinsX()+1): 
            if (histo.GetBinContent(i) == 0): 
                keys_histos_with_holes.append(key)
                break
                break
    return keys_histos_with_holes

class Quiet:
    ''' Context manager for silencing certain ROOT operations.  Usage:
    with Quiet(level = ROOT.kInfo+1):
       foo_that_makes_output

    You can set a higher or lower warning level to ignore different
    kinds of messages.  After the end of indentation, the level is set
    back to what it was previously.
    '''
    def __init__(self, level=ROOT.kError+1):
        self.level = level

    def __enter__(self):
        self.oldlevel = ROOT.gErrorIgnoreLevel
        ROOT.gErrorIgnoreLevel = self.level

    def __exit__(self, type, value, traceback):
        ROOT.gErrorIgnoreLevel = self.oldlevel