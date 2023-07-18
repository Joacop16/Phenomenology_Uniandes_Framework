from multiprocessing import cpu_count
from multiprocessing import Pool

import os
import numpy as np
import pandas as pd
from xgboost  import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

from Uniandes_Framework.ml_tools.abstract_classifier import Abstract_Classifier
from Uniandes_Framework.ml_tools import tools

DEF_PARAMETERS = {
    "n_estimators":[
        10,
        25,
        50,
        75,
        100,
        250,
        500,
        750,
        1000
    ],
    "max_depth":[
        1,
        3,
        5,
        7,
        9
    ],
    "learning_rate":[
        0.01,
        0.1,
        1,
        10,
        100
    ]
}
DEF_N_CPU=cpu_count()

class XGB_Classifier(Abstract_Classifier):
    def __init__(self,*args,**kwargs):
        self.model_name="Gradient_Boosting"
        self._n_cpu=kwargs.get("n_cpu",DEF_N_CPU)
        self._cv=kwargs.get("cv",5)
        if (self._n_cpu>DEF_N_CPU):
            print(f"The maximum number of cpus must to be: {DEF_N_CPU}")
            print(f"Setting, n_cpu = {DEF_N_CPU}")
            self._n_cpu=DEF_N_CPU
        if (self._n_cpu < 4):
            self._nthread=self._n_cpu
        else: 
            self._nthread=4
        self._parameters=kwargs.get("parameters",DEF_PARAMETERS)
        super().__init__(*args,**kwargs)
        
    def _get_good_model(self):
        
        gbc = XGBClassifier(
            objective= 'binary:logistic',
            nthread=self._nthread,
            #seed=42
        )
        
        cv = GridSearchCV(
            gbc,
            self._parameters,
            n_jobs = max(int(self._n_cpu/self._nthread)-1,1),
            cv=self._cv,
            verbose=True
        )
        cv.fit(self.trainPred,self.trainLab)
        
        self.learning_rate=cv.best_params_["learning_rate"]
        self.n_estimators=cv.best_params_["n_estimators"]
        self.max_depth=cv.best_params_["max_depth"]
        
        print("="*70)
        print(f"For the {self.model_name} model")
        print(f"the Best Parameters are {cv.best_params_}")
        return cv.best_estimator_
    
    def get_important_features(self):
        try: self.model
        except AttributeError: self.model = self._get_good_model()
        importances = self.model.feature_importances_
        features = list(self.signal_data_balanced.keys())
        ranking = np.argsort(-np.abs(importances))
        return [[features[i], importances[i]] for i in ranking] 

    def get_metrics(self, verbose = False):
        
        self.importances_df = pd.DataFrame(self.get_important_features())

        test_preds = self.model.predict(self.testPred)
        test_accuracy = accuracy_score(self.testLab,test_preds)
        
        train_preds = self.model.predict(self.trainPred)
        train_accuracy = accuracy_score(self.trainLab, train_preds)
        
        if verbose:
            print(f"the train accuracy is {train_accuracy}")
            print(f"the test accuracy is {test_accuracy}")
            print("the most important variables are")
            print(self.importances_df)
            print("="*70)
            
        return test_accuracy, train_accuracy, self.importances_df
    
    def get_most_important_features(self):
        try: 
            feat_importants = list(self.importances_df[0])
        except AttributeError:
            self.get_metrics()
            feat_importants = list(self.importances_df[0])
 
        # df_correlation = tools.dataframe_correlation(tools.concat_data([self.bkg_data_balanced, self.signal_data_balanced], balance = True))
    
        df_correlation = tools.dataframe_correlation(self.bkg_data_balanced)
        limit_correl_value = 0.6

        while len(feat_importants) > 10:
            len_i = len(feat_importants)

            for key_1 in feat_importants:
                for key_2 in df_correlation[key_1].keys():
                    if key_1 == key_2: continue
                    if not (key_2 in feat_importants): continue
                    if(abs(df_correlation[key_1][key_2]) >= limit_correl_value): 
                        if (len(feat_importants) > 10): feat_importants.remove(key_2)

            len_f = len(feat_importants)
            if (len_f == len_i): limit_correl_value = limit_correl_value - 0.01   
            
        self.feat_importants = feat_importants
        
        return feat_importants
        
    def filter_by_features(self, best_features = None, n_pca = 10, before_pca_matrix = ''):
        if not (best_features): best_features = self.get_most_important_features()
        self.bkg_data_balanced = self.bkg_data_balanced.loc[:, best_features]
        self.signal_data_balanced = self.bkg_data_balanced.loc[:, best_features]
        self.trainPred = self.trainPred.loc[:, best_features]
        self.testPred = self.testPred.loc[:, best_features]
        #First Attempt of PCA:
        
        # if before_pca_matrix != '':
        #     self.Matrix_Transformation = np.asmatrix(pd.read_csv(before_pca_matrix))
        #     self.trainPredPCA = pd.DataFrame(np.asmatrix(self.trainPred)@self.Matrix_Transformation)
        # else:           
        #     pca = PCA(n_components = n_pca)
        #     self.pca = pca.fit_transform(self.trainPred)
        #     self.Matrix_Transformation = np.linalg.pinv(np.asmatrix(self.trainPred))@pca
            
        #     self.trainPredPCA = pd.DataFrame(self.pca)
        #     self.testPredPCA = pd.DataFrame(np.asmatrix(self.testPred)@self.Matrix_Transformation)
            
        #     path_to_save = os.path.join(os.getcwd(), f"{self.model_name}.csv")
        #     pd.DataFrame(self.Matrix_Transformation).to_csv(path_to_save, index = False)