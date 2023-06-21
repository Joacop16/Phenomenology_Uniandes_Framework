import os
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import ROOT
from ROOT import TH1F
from ROOT import TCanvas #It is necessary to show plots with ROOT.
from ROOT import TH1F #It is necessary to plot histograms with ROOT.
from ROOT import THStack #It is necessary to plot many histograms at the same time with ROOT.
from ROOT import TLegend #It is necessary to plot labels when you plot many histograms at the same time with ROOT.
from ROOT import TFile #It is necessary to save histograms in a .root file.

from Uniandes_Framework.delphes_reader.particle.abstract_particle import Particle

class Quiet:
    ''' Context manager for silencing certain ROOT operations.  Usage:
    with Quiet(level = ROOT.kInfo+1):
       foo_that_makes_output

    You can set a higher or lower warning level to ignore different
    kinds of messages.  After the end of indentation, the level is set
    back to what it was previously.
    '''
    def __init__(self, level=ROOT.kError+1):
        self.oldlevel = ROOT.gErrorIgnoreLevel
        self.level = level

    def __enter__(self):
        ROOT.gErrorIgnoreLevel = self.level

    def __exit__(self, type, value, traceback):
        ROOT.gErrorIgnoreLevel = self.oldlevel



def get_kinematics_row(particles : list)->dict:
    """Extracts main kinematic variables of a particle (or more) and returns a dictionary with them.

    Parameters:
        particles (list): any number of Particle objects.

    Returns:
        dict: contains main kinematic variables.
    """
    if not isinstance(particles, list):
        raise TypeError("Particles must be a list of Particle objects")

    if any( not isinstance(particle, Particle) for particle in particles):
        #raise TypeError("Particles must be a list of Particle objects")
        pass

    row = {}
    for i, particle in enumerate(particles):

        # Save main kinematic variables
        name = particle.Name
        row[f"pT_{{{name}}}(GeV)"]=particle.pt
        row[f"#eta_{{{name}}}"]=particle.eta
        row[f"#phi_{{{name}}}"]=particle.phi
        row[f"Energy_{{{name}}}(GeV)"]=particle.energy
        row[f"Mass_{{{name}}}(GeV))"] = particle.m

        # Calculate Delta Functions with other particles
        for j in range(i+1, len(particles)):
            co_particle = particles[j]
            co_name = co_particle.Name
            row[f"#Delta{{R}}_{{{name}{co_name}}}"]=particle.DeltaR(co_particle)
            row[f"#Delta{{#eta}}_{{{name}{co_name}}}"]=particle.DeltaEta(co_particle)
            row[f"#Delta{{#phi}}_{{{name}{co_name}}}"]=particle.DeltaPhi(co_particle)
            row[f"#Delta{{pT}}_{{{name}{co_name}}}(GeV)"]=particle.sDeltaPT(co_particle)
            row[f"#Delta{{#vec{{pT}}}}_{{{name}{co_name}}}(GeV)"]=particle.vDeltaPT(co_particle)
            row[f"#Delta{{#vec{{p}}}}_{{{name}{co_name}}}(GeV)"]=particle.vDeltaP(co_particle)
    return row

default_hist_bins_dict={
    "#Delta{R}":[96,0,7],
    "#Delta{#eta}":[80,-5,5],
    "#Delta{#phi}":[52,-3.25,3.25],
    "#Delta{pT}":[120, 0.0, 1500.0],
    "#Delta{#vec{pT}}":[240, 0.0, 4800.0],
    "#Delta{#vec{p}}":[240, 0.0, 4800.0],
    "MET(GeV)":[80, 0.0, 1000.0],
    "pT_": [160, 0.0, 2000.0],
    "sT(GeV)": [200, 0.0, 4000.0],
    "mT(GeV)": [200, 0.0, 4000.0],
    "#eta_":[80, -5, 5],
    "#phi_":[128, -3.2, 3.2],
    "Energy_":[80, 0.0, 1000.0]
}

def make_histograms(df: pd.DataFrame, integral: float = 1.0, hist_bins_dict: dict = None) -> dict:
    '''Creates histograms from the data in a DataFrame using ROOT.
    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        integral (float): The desired integral of the histograms (default = 1.0).
        hist_bins_dict (dict): A dictionary containing the binning for each histogram (default = None).
    Returns:
        dict: A dictionary containing the histograms.
    '''
    if hist_bins_dict is None:
        hist_bins_dict = default_hist_bins_dict

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if not isinstance(integral, float):
        raise TypeError("integral must be a float")
    if not isinstance(hist_bins_dict, dict):
        raise TypeError("hist_bins_dict must be a dictionary")

    hist_dict = {}

    for column in df.columns:
        for hist_key, bins in hist_bins_dict.items():
            if hist_key in column:
                x_axis = column.replace('(', '[').replace(')', ']')
                hist = ROOT.TH1F(column, f"{x_axis}; Events", bins[0], bins[1], bins[2])
                hist.SetDirectory(0)
                [hist.Fill(dato) for dato in df[column]]
                hist.Scale(integral / hist.Integral() if hist.Integral() != 0 else 1.0)
                hist_dict[column] = hist
    return hist_dict

def histos_matplotlib(
        Dataset: pd.DataFrame, 
        column_key: str, 
        log: bool = False, 
        c: str = 'blue', 
        file_name: str = '', 
        nbins: int = 100
        ) -> None:
    ''' Uses matplotlib to create histograms using all data contained in a column of a DataFrame.  
    Parameters:
        Dataset (DataFrame): It is a DataFrame where each row correspond to a different particle and each column to its corresponding kinematic variable value.
        column_key (str): It is the key of the column that we want to plot as a histogram.
        log (bool): If True, the logarithm of the data will be used.
        c (str): Histogram color.
        file_name (str): File name that would be used to save the plot.
        nbins (int): Bins number.   
    '''  

    if not isinstance(Dataset, pd.DataFrame):
        raise TypeError("Dataset must be a pandas DataFrame")
    if not isinstance(column_key, str):
        raise TypeError("column_key must be a string")
    if not isinstance(log, bool):
        raise TypeError("log must be a boolean")
    if not isinstance(c, str):
        raise TypeError("c must be a string")
    if not isinstance(file_name, str):
        raise TypeError("file_name must be a string")
    if not isinstance(nbins, int):
        raise TypeError("nbins must be an integer")
    

    data = Dataset[column_key]
    if log:
        data = np.log10(data)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(data, bins=nbins, color=c, density=True)
        
    name = '$' + column_key.replace('#' , "\\" ).replace('(' , "[" ).replace(')' , "]" ) + '$'
    ax.set_xlabel(name, fontsize=12)
    ax.set_ylabel('A.U', fontsize=12)
    
    statistics = f'Mean = {data.mean():.3f}, STD = {data.std():.3f}'
    ax.set_title(statistics, loc='right', fontsize=12)
    
    if file_name:
        fig.savefig(file_name, bbox_inches='tight')
    
    plt.show()

def overlap_histos(
        kinematic_variable: str, 
        dict_histos: Dict[str, Dict[str, TH1F]], 
        alpha: float = 0.05,
        stack: bool = False, 
        log_scale: bool = False, grid: bool = False
        ) -> Tuple[THStack, TCanvas, TLegend]:
    '''Uses ROOT to overlap histograms using all kinematic variable's histograms contained in a directory.
    Parameters:
        kinematic_variable (str): Name of the kinematic variable. It must be also the key to access the corresponding histograms inside dict_histos.
        dict_histos (dict): Directory that contains all the histograms. This dictionary should have keys with the name of the signals, 
                            and each signal should have other dictionaries with the same structure as an output of make_histograms.
        alpha (float): Histogram transparency. It must be between 0 and 1.
        stack (bool): If True, the plot of histograms will consider a Stack between them.
        log_scale (bool): If True, the histogram will be plotted using log 10 scale.
        grid (bool): If True, the canvas will plot a grid in the graphic.
    
    Returns:
        tuple of THStack, TCanvas, and TLegend objects.
        
    '''

    if not isinstance(kinematic_variable, str):
        raise TypeError("kinematic_variable must be a string")
    if not isinstance(dict_histos, dict):
        raise TypeError("dict_histos must be a dictionary")
    if not isinstance(alpha, float):
        raise TypeError("alpha must be a float")
    if not isinstance(stack, bool):
        raise TypeError("stack must be a boolean")
    if not isinstance(log_scale, bool):
        raise TypeError("log_scale must be a boolean")
    if not isinstance(grid, bool):
        raise TypeError("grid must be a boolean")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    

    canvas = TCanvas('','', 600, 400)
    legend = TLegend(0.6, .8, 0.89, .89)
    legend.SetNColumns(4)
    legend.SetLineWidth(1)

    histos = THStack('hist', '')

    for i, (signal, histo_dict) in enumerate(dict_histos.items()):
        if histo_dict:
            histo = histo_dict[kinematic_variable]
            histo.SetLineColor(i + 1)
            histo.SetFillColorAlpha(i + 1, alpha)
            histo.SetLineWidth(2)
            histo.SetDirectory(0)
            histos.Add(histo)
            legend.AddEntry(histo, signal)

    x_axis = kinematic_variable.replace('(', '[').replace(')', ']')

    if stack:
        histos.Draw("hist")
        histos.SetTitle(f'; {x_axis}; Events')
    else:
        histos.Draw("histnostack")
        histos.SetTitle(f'; {x_axis}; A.U')

    if log_scale:
        canvas.SetLogy()
        #histos.SetMinimum(10)
        #histos.SetMaximum(1e8)

    if grid:
        canvas.SetGrid()

    canvas.Draw()
    legend.Draw('same')

    return histos, canvas, legend


def sum_histos(histo_list: List[TH1F], substract = False) -> TH1F:
    '''Sums histograms in a list using ROOT library.
    Parameters:
        histo_list (List[TH1F]): List of histograms to be summed.
    Return:
        TH1F: Histogram with the sum of all histograms in histo_list.
    '''
    
    # Check that histo_list is a list of TH1F
    if not all(isinstance(histo, TH1F) for histo in histo_list):
        raise TypeError("histo_list must be a list of TH1F")
    
    # Check that all histograms have the same number of bins and bin width
    bin_width = histo_list[0].GetBinWidth(0)
    nbins = histo_list[0].GetNbinsX()
    
    if any(histo.GetNbinsX() != nbins or not np.isclose(histo.GetBinWidth(0), bin_width) for histo in histo_list):
        raise ValueError("All histograms must have the same number of bins and bin width")
    
    # Initialize result histogram with bin information
    xlow = histo_list[0].GetBinLowEdge(1)
    xup = histo_list[0].GetBinLowEdge(nbins) + bin_width
    result = TH1F('sum', 'sum', nbins, xlow, xup)
    result.SetDirectory(0)
    
    if substract:
        result.Add(histo_list[0])
        for n in range(1, len(histo_list)): result.Add(histo_list[n], -1.0)
    else:
        # Sum histograms and errors
        for histo in histo_list:
            result.Add(histo)
    
    return result

def generate_csv(dictionary_list :list ,file_name: str) -> None:
    ''' Uses Pandas to create a csv file using all data contained in a list of directories.  
    Parameters:
        dictionary_list (list): It is a list where each member is a dictionary with the structure of get_kinematics_row outputs.
        file_name (string): It is the name that the .csv file will have.
    '''      
    if not all(isinstance(directory_kinematics, dict) for directory_kinematics in dictionary_list):
        raise TypeError("dictionary_list must be a list of dictionaries")
    if not isinstance(file_name, str):
        raise TypeError("file_name must be a string")
    
    Data = pd.DataFrame()

    for dictionary_kinematics in dictionary_list:
        row = pd.DataFrame.from_dict(dictionary_kinematics, orient = "index").T
        Data = pd.concat([Data,row]) 
        Data.reset_index(drop=True, inplace=True)
    Data.to_csv(file_name, index= False)
    

def save_histograms_png(path_to_save: str, dict_hist: Dict[str, TH1F], log_y: bool = False) -> None:
    """Save histograms as .png files.

    Parameters:
        path_to_save: Folder name that will be used to save all histograms as .png files.
        dict_hist: Dictionary that contains all the histograms.
        log_y: If True, the histogram will be plotted using log 10 Y-scale.
    """

    if not isinstance(path_to_save, str):
        raise TypeError("path_to_save must be a string")
    if not all(isinstance(histo, TH1F) for histo in dict_hist.values()):
        raise TypeError("dict_hist must be a dictionary of TH1F")
    if not isinstance(log_y, bool):
        raise TypeError("log_y must be a boolean")
    
    for key, histo in dict_hist.items():
        canvas = ROOT.TCanvas(key, "", 0, 0, 1280, 720)
        canvas.SetGrid()
        if log_y:
            canvas.SetLogy()
        histo.Draw("hist")
        canvas.SaveAs(os.path.join(path_to_save, f"histograms_{key}.png").replace("#", "").replace("{", "").replace("}", "").replace(" ", "_"))       
        
def write_root_file(file_name: str, dict_Hist : Dict[str, TH1F]) -> None:
    """
    This function writes a root file with the histograms contained in a dictionary.
    Parameters:
        file_name (string): It is the name that the .root file will have.
        dict_Hist (dictionary): It is a dictionary where the keys are the names of the histograms and the values are the TH1F histograms .
    """
    if not isinstance(file_name, str):
        raise TypeError("name must be a string")
    if not all(isinstance(histogram, TH1F) for histogram in dict_Hist.values()):
        raise TypeError("dict_Hist must be a dictionary of TH1F histograms")
    

    ROOT_File = TFile.Open(file_name, 'RECREATE') 
    
    [dict_Hist[key].Write() for key in dict_Hist.keys()]

    ROOT_File.Close()
    
def read_root_file(path_root_file: str, expected_keys: list) -> dict:
    """
    This function reads a root file and returns a dictionary with the histograms contained in the root file.
    Parameters:
        path_root_file (string): It is the path of the root file.
        expected_keys (list): It is a list with the names of the histograms that are expected to be in the root file.
        
    Returns:
        dictionary: It is a dictionary where the keys are the names of the histograms and the values are the histograms.
    """
    Dict_hist = {}
    File = TFile.TFile.Open(path_root_file, 'READ')
    for key in expected_keys:
        histogram = File.Get(key)
        try: histogram.SetDirectory(0)
        except: pass
        Dict_hist[key] = histogram
    File.Close()
    return Dict_hist
               
def review_holes_in_histograms(Dict_Hist: Dict[str, TH1F]) -> List[str]:
    """
    Returns a list with the names of all histograms with holes contained in a python dictionary (Dict_Hist).
    Parameters:
        Dict_Hist (Dict[str, TH1F]): It is the dictionary that contains all the histograms.
    Return:
        List[str]: List with the names of all histograms with holes.
    """

    if not all(isinstance(histogram, TH1F) for histogram in Dict_Hist.values()):
        raise TypeError("Dict_Hist must be a dictionary of TH1F histograms")
    
    keys_histos_with_holes = []
    for key, histo in Dict_Hist.items():
        if any(histo.GetBinContent(i) == 0 for i in range(1, histo.GetNbinsX()+1)):
            keys_histos_with_holes.append(key)
    return keys_histos_with_holes


def write_txt_file_with_high_per_bin(file_name :str, Dict_Hist :Dict[str, TH1F]) -> None:
    """
    This function writes a txt file with the number of events per bin of each histogram contained in a dictionary.
    Parameters:
        name (string): It is the name that the .txt file will have.
        Dict_Hist (dictionary): It is a dictionary where the keys are the names of the histograms and the values are the histograms.
    """
    for key in Dict_Hist.keys():
        histo = Dict_Hist[key]
        high_list = [histo.GetBinContent(i) for i in range(1, histo.GetNbinsX())]
        txt_name = f'{file_name}_{key}.dat'
        np.savetxt(txt_name.replace('#', '').replace('{', '').replace('}', '').replace(' ', '_'), high_list)