import numpy as np
import pandas as pd
import shap
import time
from tqdm.notebook import tqdm
from pySAR.pySAR import *
pys = PySAR('data.json')
pys.encode_aai(indices='ARGP820101')

def process(values,aai,desc2):
    elements=[]
    row_names=[]
    for i in range(len(values)):
        value=values[i]
        elements.append(value)
        if i<298: # "298" represents the dimensions of the AAindex feature vector.
            row_name=aai+'_'+str(i+1)
            row_names.append(row_name)
        if 297<i<298+20: # "20" represents the dimensions of the first descriptor feature vector.
            row_name=desc2[0]+'_'+str(i+1-298)
            row_names.append(row_name)
        if 297+20<i:
            row_name=desc2[1]+'_'+str(i+1-298-20)
            row_names.append(row_name)
    new_values=pd.DataFrame({'Index_'+aai+desc2[0]+desc2[1]:row_names,aai+desc2[0]+desc2[1]:elements})
    return new_values

algorithm =  ['randomforestregressor']
all_indices =['ARGP820101']
comb_list=[['amino_acid_composition', 'conjoint_triad']]


for alg in algorithm:
    start = time.time()
    pys.model.algorithm = alg
    pys.algorithm = alg
    print(pys.model.get_model())

    tqdm_disable = False
    if (len(all_indices)) <= 1:
        tqdm_disable = True

    count=0
    for index in tqdm(all_indices, unit="aai+2desc", position=0,
                desc="aai+2desc", mininterval=30, disable=tqdm_disable):
        for desc2 in comb_list:
                print(index,desc2)

                df1 = pys.encode_aai_descriptor(indices=index,descriptors=desc2,show_plot=False, print_results=False)
                model=pys.model.get_model()
                dic=pys.model.model.get_params()
                model.set_params(**dic)
                model.fit(pys.model.X, pys.model.Y.ravel())

                explainer = shap.Explainer(model.predict,pys.model.X)
                shap_values = explainer.shap_values(pys.model.X)
                col_name=index+desc2[0]+desc2[1]
                np.save('data/Shap/shapvalues_original/{}_{}'.format(alg, col_name),shap_values)

                shap_values=shap_values.mean(0)
                shap_values=process(shap_values,index,desc2)
                print(shap_values)
                shap_values.to_csv('data/Shap/shapvalues/{}_{}.csv'.format(alg, col_name))


def shap_sort(path,index,AAI):
    dfshap=pd.read_csv(path)

    temp=[]
    temp.append('Index_'+AAI+index[0]+index[1])
    temp.append(AAI+index[0]+index[1])

    df_encodingway=pd.DataFrame(columns=temp)
    df_encodingway[AAI+index[0]+index[1]]=dfshap.loc[:,AAI+index[0]+index[1]].abs().tolist()
    df_encodingway['Index_'+AAI+index[0]+index[1]]=dfshap.loc[:,'Index_'+AAI+index[0]+index[1]]
    df_encodingway=df_encodingway.sort_values(by=AAI+index[0]+index[1] , inplace=False,ascending=False)
    df_encodingway=df_encodingway.reset_index(drop=True)
    path='data/Shap/shapvalues_rank/rank_{}_{}.csv'.format(alg, col_name)
    df_encodingway.to_csv(path)

path='data/Shap/shapvalues/{}_{}.csv'.format(alg, col_name)
all_indices ='ARGP820101'
methods=['amino_acid_composition', 'conjoint_triad']

shap_sort(path,methods,all_indices)