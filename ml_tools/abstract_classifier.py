import os
from abc import ABC
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
import Uniandes_Framework.ml_tools.tools as tools

class Abstract_Classifier(ABC):
    def __init__(self,*args,**kwargs):
        
        self.signal_dictionary = kwargs.get('signal_dictionary')
        self.bkg_dictionary = kwargs.get('bkg_dictionary')
        self.all_signal_dictionary = self.signal_dictionary | self.bkg_dictionary
        
        # self.signal_path_csv_list = [i[1] for i in self.signal_dictionary.items()]
        # self.bkg_path_csv_list = [i[1] for i in self.bkg_dictionary.items()]
    
        balance = kwargs.get('balance', True)
        self._prepare_data(balance)
    
    def _prepare_data(self, balanced = True):
        
        bkg_dataframe_list = [tools.concat_signals(self.bkg_dictionary[key], balance = balanced) for key in self.bkg_dictionary]
        signal_dataframe_list = [tools.concat_signals(self.signal_dictionary[key], balance = balanced) for key in self.signal_dictionary]
        
        self.signal_data_balanced = tools.concat_data(signal_dataframe_list, balance = balanced)
        self.bkg_data_balanced = tools.concat_data(bkg_dataframe_list, balance = balanced)
                
        X , Y = tools.prepare_to_train(self.signal_data_balanced,self.bkg_data_balanced, balance = True)
        self.trainPred, self.testPred, self.trainLab, self.testLab = train_test_split(X, Y, test_size=0.20)
        
    def fit_model(self):
        self.model.fit(self.trainPred, self.trainLab)
        
    def save_model(self, path_to_save = os.getcwd(), file_name = None):
        if not file_name: file_name = f"{self.model_name}.joblib"
        file = os.path.join(path_to_save, file_name)
        joblib.dump(self.model, file)
        print(f'The model was saved in: \n{file}')
        
    def set_name_model(self, name):
        self.model_name = name
            


        