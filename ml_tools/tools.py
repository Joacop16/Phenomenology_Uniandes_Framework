import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ROOT import kBlue, kRed, kBlack
from ROOT import TCanvas, TH1F, TFile
import uproot

import joblib
import os

from Uniandes_Framework.delphes_reader.root_analysis import Write_ROOT_File, Write_txt_file_with_high_per_bin

def concat_signals(path_csv_list, balance = True):
    ''' Read a list of Dataframe's paths and concatenate them in a Bigger Dataframe that contains all data.
    Parameters:
        path_csv_list (string list): list with paths of all Dataframes that will be concatenated.
        balance (Boolean): Boolean that says if it is necessary to balance all dataframes.
    Return:
        Pandas DataFrame: Bigger Dataframe with all data.
    '''  
    Datas = {}
    for path in path_csv_list:
        Datas[path] = pd.read_csv(path, index_col = 0)
        Datas[path] = Datas[path].sample(frac = 1) #mix Datas[path] rows
        
    keys = list(Datas.keys())
    sizes = [len(Datas[key]) for key in keys]
    if balance: num_rows_per_df = min(sizes)
    else: num_rows_per_df = max(sizes)
            
    Data = Datas[keys[0]].head(num_rows_per_df)
    for i in range(1, len(keys)): Data = pd.concat([Data, Datas[keys[i]].head(num_rows_per_df)], axis = 0)
    
    Data = Data.dropna() #Delete rows with nan values
    Data = Data.sample(frac = 1) #mix Datas[path] rows
    Data.reset_index(drop=True, inplace=True)    
    
    return Data

def prepare_to_train(Signal_DataFrame, BKG_DataFrame, balance = True, verbose = False):
    ''' Create X and Y DataFrames to train Machine Learning Algorithmes using a Signal DataFrame and BKG DataFrame.
    Parameters:
        Signal_DataFrame (Pandas DataFrame): Dataframe with signal data.
        BKG_DataFrame (Pandas DataFrame): Dataframe with bkg data.
        balance (Boolean): Boolean that says if it is necessary to balance signal and bkg dataframes.
        verbose (Boolean): Boolean that says if it is necessary to print the quantity of signal and background.
    Return:
        Pandas DataFrame: X and Y to train Machine Learning Algorithmes.
    '''
    if balance == True: num_rows_per_df = min(len(Signal_DataFrame), len(BKG_DataFrame))
    else: num_rows_per_df = max(len(Signal_DataFrame), len(BKG_DataFrame))
    
    Label_column = pd.DataFrame.from_dict({'Label': [1 for i in range(num_rows_per_df)]})
    
    Signal_DataFrame = pd.concat([Signal_DataFrame.loc[:num_rows_per_df], Label_column], axis = 1)
    BKG_DataFrame = pd.concat([BKG_DataFrame.loc[:num_rows_per_df], Label_column*0], axis = 1)    

    Data = pd.concat([Signal_DataFrame, BKG_DataFrame], axis = 0)
    Data = Data.dropna() #Delete rows with nan values
    Data = Data.sample(frac = 1) #mix Datas[path] rows
    Data.reset_index(drop=True, inplace=True)
    
    X = Data.loc[:, Data.columns!= 'Label']
    Y = Data.loc[:, 'Label']   
    
    if verbose: print(f'There are {len(Signal_DataFrame)} events of signal, and {len(BKG_DataFrame)} of background.')
    
    return X, Y

def dataframe_correlation(DataFrame, plot = False, pdf_name = ''):
    ''' Return and plot (optional) the correlation between all columns in a DataFrame.
    Parameters:
        Data* (Pandas DataFrame): Dataframe to be analized.
        plot (Boolean): Boolean that says if the user want to plot the correlation as a heatmap.
        pdf_name (String): Name that we will be used to save the plot. It must include the format (.pdf or .png).
    Return:
        Pandas DataFrame: Dataframe with the correlation.
    '''  
    Data = DataFrame.copy()
    if plot:
        columns=[]
        for column in Data.columns:
            if (column == "light_jets_multiplicity"): columns.append(r"$n_j$")
            else: 
                name = column.replace("#","\\").replace("(GeV)", "").replace("$", "")
                columns.append(fr'${name}$')   
        Data.columns = columns
        Correlation = Data.corr()

        fig = plt.figure(figsize = (12, 9))
        sns.heatmap(Correlation, xticklabels=True, yticklabels=True)
        if (pdf_name != ''): plt.savefig(pdf_name, bbox_inches='tight')
        
    else: Correlation = Data.corr()
    
    return Correlation

def hist_discriminator(path_model, csv_dict, path_to_save = '', best_features = []):
    ''' Return a directory with machine learning discriminator histograms following the same structure that csv_dict. 
    Parameters:
        path_model (string): Path of the machine learning model that will be used to plot the histograms.
        csv_dict (Python dictionary): dictionary with the name of the signal an its respectively csv file path.
        path_to_save (String): Path of the folder that we will be used to save all the histograms.
    Return:
        TH1F (Python dictionary): Directory with machine learning discriminator histograms. 
    '''
    model = joblib.load(open(path_model, 'rb'))
    name = os.path.basename(path_model)
    name = name.split('.')[0]
    
    histos = {}
    for key in csv_dict.keys():
    
        histo = TH1F(f"{key}",f"{key};ML-score;nevents(137/fb)", 100, 0.0,1.0)
        histo.SetLineWidth(1)
        histo.SetDirectory(0)
        histo.SetLineColor(kBlue)
        histo.SetFillColor(kBlue)
        
        data_to_evaluate = pd.read_csv(csv_dict[key], index_col = 0)
        if (len(best_features) != 0): data_to_evaluate = data_to_evaluate.loc[:, best_features]
        
        scores = model.predict_proba(data_to_evaluate)[:,1]
        Data =  pd.DataFrame(scores, columns = ['scores'])

        for score in scores:
            histo.Fill(score)
        histo.Scale(1/histo.Integral())
        
        histos[key] = histo

        if (path_to_save != ''):
            c1 = TCanvas( f'c-{key}', '', 0, 0, 1280, 720)
            c1.SetGrid()
            c1.SetLogy()
            histo.Draw("hist")
            Data.to_csv(os.path.join(path_to_save,f"{name}_{key}.csv"),index=False)
            c1.SaveAs(os.path.join(path_to_save,f"{name}_{key}.png"))
            
    if (path_to_save != ''): 
        Write_ROOT_File(os.path.join(path_to_save,f"Histograms_{name}.root"), histos)
        Write_txt_file_with_high_per_bin(os.path.join(path_to_save,f"{name}"), histos)
        
    return histos    