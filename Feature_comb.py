import time
import os
import logging
import matplotlib.pyplot as plt
import itertools
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from scipy.stats import spearmanr
from multiprocessing import Pool
from pySAR.pySAR import *

logging.basicConfig(filename='debug.log', level=logging.DEBUG, 
                    format='%(asctime)s %(levelname)s:%(message)s')

def get_AAindex_2Desc(name, N):
    pys = PySAR('data.json')
    pys.encode_descriptor(descriptor='amino_acid_composition')
    valid_descriptors = [
            'amino_acid_composition', 
            'dipeptide_composition', 
            'conjoint_triad',
            'geary_autocorrelation',
            'sequence_order_coupling_number',
            'quasi_sequence_order'
    ]
    algorithm = [
            'randomforestregressor', 
            'gradientboostingregressor',
            'xgb',
            'adaboostregressor', 
            'baggingregressor', 
            'plsregression',
            'linearregression',  
            'decisiontreeregressor', 
            'kneighborsregressor', 
            'svr'
    ]
    all_indices = aaindex1.record_codes()
    comb_list = [list(x) for x in list(itertools.combinations(valid_descriptors, 2))]
    for alg in algorithm:
        start = time.time()
        pys.model.algorithm = alg
        pys.algorithm = alg

        tqdm_disable = False
        if len(all_indices) <= 1:
            tqdm_disable = True
        alg_dir = os.path.join('data', 'Feature_comb', alg)
        if not os.path.exists(alg_dir):
            os.makedirs(alg_dir)
        index = all_indices[N]
        df = pd.DataFrame(columns=['algorithm', 'Index', 'Category', 'Descriptor', 'Group', 'R2', 'RMSE', 'MSE', 'MAE', 'Spearman'])
        output_file = os.path.join(alg_dir, f'{index}_2desc.csv')
        if os.path.exists(output_file):
            print(f'{index} already calculated!')
            continue
        for index2 in comb_list:
            print(alg,index,index2)
            logging.info(f"Calling encode_aai_descriptor with index: {index}, descriptors: {index2}")
            try:
                df1 = pys.encode_aai_descriptor(indices=index, descriptors=index2, show_plot=False, print_results=False)
            except Exception as e:
                logging.error(f'Error in encode_aai_descriptor for {index} with descriptors {index2}: {e}')
                continue
            
            if df1 is None:
                logging.warning(f'encode_aai_descriptor returned None for {index} with descriptors {index2}')
                continue
            if df1.empty:
                logging.warning(f'encode_aai_descriptor returned empty DataFrame for {index} with descriptors {index2}')
                continue
            
            def spearman_corr_score(estimator, X, y):
                y_pred = estimator.predict(X)
                return spearmanr(y, y_pred).correlation
            
            ten_fold = KFold(n_splits=10, shuffle=True, random_state=0)
            try:
                ten_fold_rr = cross_val_score(pys.model.get_model(), pys.model.X, pys.model.Y.ravel(), cv=ten_fold, scoring='r2')
                ten_fold_rmse = cross_val_score(pys.model.get_model(), pys.model.X, pys.model.Y.ravel(), cv=ten_fold, scoring='neg_root_mean_squared_error')
                ten_fold_rmse = -ten_fold_rmse
                ten_fold_mse = cross_val_score(pys.model.get_model(), pys.model.X, pys.model.Y.ravel(), cv=ten_fold, scoring='neg_mean_squared_error')
                ten_fold_mse = -ten_fold_mse
                ten_fold_mae = cross_val_score(pys.model.get_model(), pys.model.X, pys.model.Y.ravel(), cv=ten_fold, scoring='neg_mean_absolute_error')
                ten_fold_mae = -ten_fold_mae
                ten_fold_scores_spearman = cross_val_score(pys.model.get_model(), pys.model.X, pys.model.Y.ravel(), cv=ten_fold, scoring=spearman_corr_score)
            except Exception as e:
                logging.error(f'Error during cross-validation for {index} with descriptors {index2}: {e}')
                continue
            
            df1['algorithm'] = alg
            df1['R2'] = ten_fold_rr.mean()
            df1['RMSE'] = ten_fold_rmse.mean()
            df1['MSE'] = ten_fold_mse.mean()
            df1['MAE'] = ten_fold_mae.mean()
            df1['Spearman'] = ten_fold_scores_spearman.mean()
            df = pd.concat([df, df1], join='inner')
            plt.cla()
            plt.close()

        if df.empty:
            logging.warning(f'No results for {index}. Skipping saving.')
            continue

        df1 = df.sort_values(by=['R2'], ascending=False)
        try:
            df1.to_csv(output_file, index=False)
            logging.info(f'Results saved to {output_file}')
        except Exception as e:
            logging.error(f'Error saving results to {output_file}: {e}')

        end = time.time()
        elapsed = end - start
        print(f'\nElapsed Time for AAI Encoding: {elapsed:.3f} seconds.')
        print(f'Process id running on {name} = {os.getpid()}; running time = {elapsed}')

if __name__ == '__main__':
    print(f'Process id = {os.getpid()}.')
    start_time = time.perf_counter()

    po = Pool(2)
    for i in range(0, 4):
        po.apply_async(get_AAindex_2Desc, ('job' + str(i), i))

    print("-----start-----")
    po.close()
    po.join()
    print("-----stop-----")
