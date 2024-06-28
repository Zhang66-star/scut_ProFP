import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from mlxtend.feature_selection import SequentialFeatureSelector
from pySAR.pySAR import *
pys = PySAR('data.json')
pys.encode_descriptor(descriptor='amino_acid_composition')

pys.model.algorithm = 'randomforestregressor'
pys.algorithm = 'randomforestregressor'
index='ARGP820101'
desc=['amino_acid_composition', 'conjoint_triad']

pys.encode_aai_descriptor(indices=index,descriptors=desc,show_plot=False, print_results=False)
col_name=index+desc[0]+desc[1]

pys.model.algorithm = 'randomforestregressor'
pys.algorithm = 'randomforestregressor'
pys.model.X.shape[1]
ten_fold = KFold(n_splits=10,shuffle=True,random_state=0)
model = pys.model.get_model()

path='data/Shap/shapvalues_rank/rank_{}_{}.csv'.format(pys.algorithm, col_name)
features =pd.read_csv(path)
                
feature_use=features.loc[:,'Index_'+index+desc[0]+desc[1]]
feature=feature_use 

data=[]
indexss=[]
for i in feature:
            if i[0]=='A': # "A" represents the initials of the AAIndex used.
                temp=int(i[11:]) # '11' indicates the number of characters in the AAIndex string plus 1
                indexss.append(temp)
            if i[0]=='a': # "a" represents the initials of the first descriptor used.
                temp=int(i[23:]) # '23' indicates the number of characters in the first descriptor string plus 1
                indexss.append(temp+298)
            if i[0]=='c': # "c" represents the initials of the second descriptor used.
                temp=int(i[15:]) # '15' indicates the number of characters in the second descriptor string plus 1
                indexss.append(temp+298+20)

indexs1=[]
for i in indexss:
            i=i-1
            indexs1.append(i)
feature_counts = []

indexs1=indexs1[:300] # Select the number of features to SFS
X=pys.model.X
X=pd.DataFrame(X)
X=X.iloc[:,indexs1]
# Definition of Spearman's correlation coefficient as an assessment indicator
# def spearman_corr_score(estimator, X, y):
#     y_pred = estimator.predict(X)
#     return spearmanr(y, y_pred).correlation
sfs = SequentialFeatureSelector(model,
                                k_features=X.shape[1],
                                forward=True,
                                floating=False,
                                verbose=1,
                                n_jobs=-1,
                                scoring='r2', #scoring=spearman_corr_score Can be replaced with Spearman's correlation coefficient as an assessment indicator
                                cv=ten_fold)
sfs = sfs.fit(X, pys.model.Y.ravel())

for k in range(1, X.shape[1] + 1):
    feature_counts.append(k)

df = pd.DataFrame.from_dict(sfs.get_metric_dict()).T
df_sorted = df.sort_values(by='avg_score', ascending=False)
df_sorted.to_csv('data/Shap/shap_SFS/{}_{}.csv'.format(pys.algorithm,col_name),index=False)