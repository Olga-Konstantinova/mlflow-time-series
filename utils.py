import numpy as np
import pandas as pd


def get_smape(df:pd.DataFrame, model:str='AutoARIMA', error_per_id_per_cutoff:bool=False, error_per_id:bool=False, overall_error:bool=True):
    """This function computes smape on different levels when specified which one is expected

    Parameters
    ----------
    df : pd.DataFrame
        dataframe that contains the predicted by model values and the 'y' column
    model : str, optional
        name of the column where the predicted values are stored, by default 'AutoARIMA'
    error_per_id_per_cutoff : bool, optional
        if True, the smape is computed per each cfip and each cutoff, by default False
    error_per_id : bool, optional
        if True, the smape is computed per each cfip, by default False
    overall_error : bool, optional
        if True, the smape across all cfip is computed, by default True

    Returns
    -------
    pd.DataFrame (s) or/and float (depending on the parameters error_per_id_per_cutoff, error_per_id, overall_error)
        the dataframe (s) with smape values or/and overall smape value
    """
    
    df['error'] = np.nan_to_num(np.abs(df['y'] - df[model]) * 2 / (np.abs(df[model]) + np.abs(df['y'])), nan=0.0)
    df['unique_id'] = df.index
    df = df.reset_index(drop=True)

    if error_per_id_per_cutoff:
        # to get error per unique id, per cutoff
        return df.groupby(['unique_id', 'cutoff']).agg({'error': 'mean'}) * 100

    if error_per_id:
        # to get error per unique id
        return df.groupby(['unique_id']).agg({'error': 'mean'}) * 100

    if overall_error:
        # get mean over all unique id's
        return df.error.mean() * 100
