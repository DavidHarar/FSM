import pandas as pd
import numpy as np
from mrmr import mrmr_classif
from datetime import datetime
import mifs
import scipy.spatial as ss
from scipy.special import digamma
from math import log
import numpy.random as nr
import numpy as np
import random
import skrebate
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from scipy import stats
from utils import *

class FSM:

    import os

    ## Note: these functions have to be download directly from the writers' github repositories.
    ## Bellow an example of how to install a package once it was downloaded.
    # Scikit Feature
    # os.chdir('<your path here>/scikit-feature-master/')
    # !python setup.py install
    # # NPEET installation
    # os.chdir('<your path here>/NPEET-master/')
    # !python setup.py install
    ## Install mifs package (not mifs method)
    ## prequisite (that for a reason couldn't be installed inside the mif's pip install)
    # !pip install 'bottleneck>=1.0.0'
    # os.chdir('/Users/davidharar/Downloads/mifs-master/')
    # !python setup.py install
    
    
    # !pip install mrmr_selection
    # !pip install skrebate        
    
    def __init__(self, k, filler):

        import multiprocessing

        cpus = multiprocessing.cpu_count()
        use = cpus-1

        self.k = k
        self.filler = filler

        self.anova_filter = SelectKBest(f_classif, k=k)
        self.chi2_filter = SelectKBest(chi2, k=k)
        self.ReliefF_selector = skrebate.ReliefF(n_features_to_select=k, n_neighbors=100, verbose=True)
        self.SURF_selector = skrebate.SURF(n_features_to_select=k, verbose=True)
        self.SURFstar_selector = skrebate.SURFstar(n_features_to_select=k, verbose=True)
        self.MultiSURF_selector = skrebate.MultiSURF(n_features_to_select=k, verbose=True)
        self.jmi = mifs.MutualInformationFeatureSelector(method = "JMI", n_jobs=use, n_features=k)
        self.jmim = mifs.MutualInformationFeatureSelector(method = "JMIM", n_jobs=use, n_features=k)
        
    def anova_inference(self, X, y, process=True):
        """
        Return K best features using sklearn's ANOVA feature selection.
        Inputs: 
        - X: a pd.DataFrame where features are on the columns.
        - y: a pd.Series of the outcome vector.
        """
        
        # get selector
        anova_filter = self.anova_filter
        filler = self.filler
        
        if process:
            # Fill na
            X = X.fillna(filler)
        
        anova_filter.fit(X, y)
        
        return anova_filter.get_feature_names_out()
        

    def kruskal_inference(self, X, y):
        """
        Return the K features with the highest Kruskal–Wallis rank-sum statistic.
        Inputs: 
        - X: a pd.DataFrame where features are on the columns.
        - y: a pd.Series of the outcome vector.
        """

        k = self.k

        kruskal_df = dict({'idx':[], 'val':[]})
        for j in range(X.shape[1]):
            kruskal_df['idx'].append(j)
            kruskal_df['val'].append(stats.kruskal(X.iloc[:,j].values, y.values)[0])
        kruskal_df = pd.DataFrame(kruskal_df)
        kruskal_df.sort_values(['val'], inplace = True)
        kruskal_df.reset_index(drop = True, inplace = True)
        
        return X.columns[kruskal_df.idx[:k]]
        
    def chi2(self, X, y, process=True):
        # get selector
        chi2_filter = self.chi2_filter
        filler = self.filler

        if process:
            # Fill na
            X = X.fillna(filler)

            # Normalize (chi2 takes only non-negative data)
            X = X - X.min(axis = 0)


        chi2_filter.fit(X, y)
        return chi2_filter.get_feature_names_out()

    def cmim(self, X,y, process=True):

        k = self.k
        filler = self.filler

        if process:
            # Fill na
            X = X.fillna(filler)        
            
            # discretize
            for v in X.columns:
                X[v] = X[v].astype('int')

        ## Convert to Numpy
        X_np = X.values
        y_np = y.values
        
        cmim_res = cmim(X_np, y_np,n_selected_features=k)
        return X.columns[cmim_res[0]]
    
    def disr(self, X,y, process=True):
        
        k = self.k
        filler = self.filler
        
        if process:
            # Fill na
            X = X.fillna(filler)        
            
            # discretize
            for v in X.columns:
                X[v] = X[v].astype('int')

        ## Convert to Numpy
        X_np = X.values
        y_np = y.values

        disr_res = disr(X_np,y_np,n_selected_features=k)
        return X.columns[disr_res[0]]

    def mifs(self, X,y, process=True):
        
        k = self.k
        filler = self.filler
        
        if process:
            # Fill na
            X = X.fillna(filler)        
            
            # discretize
            for v in X.columns:
                X[v] = X[v].astype('int')

        ## Convert to Numpy
        X_np = X.values
        y_np = y.values

        mifs_res = mifs_(X_np,y_np,n_selected_features=k)
        return X.columns[mifs_res[0]]
        
    def fcbf(self, X,y, process=True):
        
        k = self.k
        filler = self.filler
        
        if process:
            # Fill na
            X = X.fillna(-1)        
            
            # discretize
            for v in X.columns:
                X[v] = X[v].astype('int')

        ## Convert to Numpy
        X_np = X.values

        fcbf_res = fcbf(X_np,y)
        return X.columns[fcbf_res[0]]
    

    def jmi(self, X, y, process=True):

        # Get selector
        jmi = self.jmi
        filler = self.filler

        if process:
            # Fill na
            X = X.fillna(filler)        

        jmi.fit(X, y)

        return X.columns[jmi.ranking_]
        
    def jmim(self, X, y, process=True):

        # Get selector
        jmim = self.jmim
        filler = self.filler

        if process:
            # Fill na
            X = X.fillna(filler)        

        jmim.fit(X, y)

        return X.columns[jmim.ranking_]
        
    def ReliefF(self, X, y):

        # Get selector
        ReliefF_selector = self.ReliefF_selector
        reliefF_res = ReliefF_selector.fit(X.values, y.values)
        return X.columns[reliefF_res.top_features_]

    def SURF(self, X, y):

        # Get selector
        SURF_selector = self.SURF_selector
        SURF_res = SURF_selector.fit(X.values, y.values)

        return X.columns[SURF_res.top_features_]

    def SURFstar(self, X, y):

        # Get selector
        SURFstar_selector = self.SURFstar_selector
        SURFstar_res = SURFstar_selector.fit(X.values, y.values)
        
        return X.columns[SURFstar_res.top_features_]

    def MultiSURF(self, X, y):

        # Get selector
        MultiSURF_selector = self.MultiSURF_selector
        MultiSURF_res = MultiSURF_selector.fit(X.values, y.values)
        
        return X.columns[MultiSURF_res.top_features_]
        
    
    def inference(self, X, y):
        """
        Feature selection for X
        Inputs:
        - X: A pandas dataframe of shape N,P where P is the number
            of features and N is the number of observations.
        - y: A pd.Series containing the binary outcome, of shape (N,).
        - B: The number of bootstrap subset to sample.
        - Sample_size: The sample size of each sample.
        Output:
        - 
        """

        k = self.k
        filler = self.filler

        # processing:
        
        ## Filled-na version
        X_filled = X.fillna(filler)
        y = y.copy()
        
        ## Discretized version
        X_disc = X_filled.copy()

        for v in X.columns:
            X_disc[v] = X_disc[v].astype('int')

        ## Nonegative version
        X_nonegative = X_filled - X_filled.min(axis = 0)

        features_anova     = self.anova_inference(X_filled, y, process=False)
        features_chi2      = self.chi2(X_nonegative, y, process=False)
        features_kruskal   = self.kruskal_inference(X, y)
        features_cmim      = self.cmim(X_disc,y, process=False)
        features_disr      = self.disr(X_disc,y, process=False)
        features_mifs      = self.mifs(X_disc,y, process=False)
        features_fcbf      = self.fcbf(X_disc,y, process=False)
        features_ReliefF   = self.ReliefF(X, y)
        features_SURF      = self.SURF(X, y)
        features_SURFstar  = self.SURFstar(X, y)
        features_MultiSURF = self.MultiSURF(X, y)

        # These three are currently unstable
        # features_mrmr
        # jmi(self, X, y, process=False)
        # jmim(self, X, y, process=False)

        results = dict({
            "ANOVA" : features_anova,
            "chi2" : features_chi2, 
            "kruskal" : features_kruskal, 
            "cmim" : features_cmim, 
            "disr" : features_disr, 
            "mifs" : features_mifs, 
            "fcbf" : features_fcbf, 
            "mrmr" : None, # features_mrmr
            "jmi" : None, # features_jmi,
            "jmim" : None, # features_jmim,
            "ReliefF" : features_ReliefF,
            "SURF" : features_SURF,
            "SURFstar" : features_SURFstar, # features_SURFstar,
            "MultiSURF" : features_MultiSURF # features_MultiSURF
            })
        
        return results

    def Bootstrapper(self, X, y, B, Sample_size):
        """
        Feature selection for X
        Inputs:
        - X: A pandas dataframe of shape N,P where P is the number
            of features and N is the number of observations.
        - y: A pd.Series containing the binary outcome, of shape (N,).
        - B: The number of bootstrap subset to sample.
        - Sample_size: The sample size of each sample.
        Output:
        - 
        """
        from tqdm import tqdm

        results = dict({
            "ANOVA" : [],
            "chi2" : [], 
            "kruskal" : [], 
            "cmim" : [], 
            "disr" : [], 
            "mifs" : [], 
            "fcbf" : [], 
            "mrmr" : [], # features_mrmr
            "jmi" : [], # features_jmi,
            "jmim" : [], # features_jmim,
            "ReliefF" : [],
            "SURF" : [],
            "SURFstar" : [], # features_SURFstar,
            "MultiSURF" : [] # features_MultiSURF
            })
        
        df = X.copy()
        df['outcome'] = y

        for b in tqdm(range(B)):
            df_sample = df.sample(n=Sample_size)
            X_sample = df_sample.drop('outcome', axis = 1)
            y_sample = df_sample['outcome']

            results_sample = self.inference(X_sample, y_sample)

            for selector in results:
                results[selector].append(results_sample[selector])

        return results















