import os
import pickle
import pandas as pd
import torch.nn as nn
import torch
from train.callbacks import History


# We will start by saving the model
def save_model(model, path):
    '''
    Save the model to the given path
    '''
    torch.save(model.state_dict(), path)

def save_model_details(model_description, data_description, hyperparameters, path):
    '''
    Save the model description to the given path
    '''
    model_path = os.path.join(path, 'model', model_description['id'] + '.pkl')
    data_path = os.path.join(path, 'data', model_description['id'] + '.pkl')
    hyp_path = os.path.join(path, 'hparam', model_description['id'] + '.pkl')
    with open(model_path, 'wb') as file:
        pickle.dump(model_description, file)
    with open(data_path, 'wb') as file:
        pickle.dump(data_description, file)
    with open(hyp_path, 'wb') as file:
        pickle.dump(hyperparameters, file)
    return model_path, data_path, hyp_path




def df_entry(model_description, data_description, history, hyperparameters, path):
    '''
    Return a pd series with the data to be stored
    '''
    model_path, data_path, hyp_path = save_model_details(model_description, data_description, hyperparameters, path)
    res = {
        'Model Name': model_description['name'],
        # 'Model ID': model_description['id'],
        'Data ID': data_description['id'],
        'hyperparamters des': hyp_path,
        'model des': model_path,
        'data des': data_path
        }
    for key, value in history.items():
        res[key] = value

    return pd.Series(res)

def update_csv(path, model_description, data_description, history, hyperparameters):
    '''
    Update the csv file with the new data
    If the csv file does not exist, it creates a new one
    '''
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        df = pd.DataFrame()
    data = df_entry(model_description, data_description, history, hyperparameters, path)
    df = df.append(data, ignore_index=True)
    df.to_csv(path, index=False)
