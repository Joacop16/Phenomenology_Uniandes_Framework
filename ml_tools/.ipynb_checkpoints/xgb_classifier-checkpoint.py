from multiprocessing import cpu_count
from multiprocessing import Pool

import pandas as pd
from xgboost  import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from ml_tools.abstract_classifier import Abstract_Classifier
from ml_tools.bc_model import BC_Model

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
    
    def get_good_params(self):
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
        print(f"For the {self.signal_names} model")
        print(f"the Best Parameters are {cv.best_params_}")
        return cv.best_params_
    
    def fit_model(self):
        self.get_good_params()
        self.model = BC_Model(
            make_pipeline(
                StandardScaler(), 
                XGBClassifier(
                    objective= 'binary:logistic',
                    learning_rate=self.learning_rate,
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    nthread=max(self._n_cpu,1),
                    use_label_encoder=False
                )
            )
        )
        self.model.fit(self.trainPred, self.trainLab)
        
        test_preds = self.model.predict(self.testPred)
        test_accuracy = accuracy_score(self.testLab,test_preds)
        
        train_preds = self.model.predict(self.trainPred)
        train_accuracy = accuracy_score(self.trainLab, train_preds)
        self.importances_df=pd.DataFrame(
            self.model.sorted_feature_importance(self.features)
        )
        print(f"the train accuracy is {train_accuracy}")
        print(f"the test accuracy is {test_accuracy}")
        print("the most important variables are")
        print(self.importances_df)
        print("="*70)