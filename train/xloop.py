from torchsummary import summary
import os
import torch
from train.train import train, evaluate
from train.callbacks import History, def_metrics
from datasets.dataset import Dataset
from torch.utils.data import DataLoader
from train.misc import EarlyStopping
from train.store import update_csv

# def evaluate(model, val_loader, criterion, device, metrics, history):

def _get_modelsummary(model, input_size):
    try:
        model_summary = summary(model, input_size)
    except:
        model_summary = model.__str__()
    return model_summary

def _get_datasummary(datapath, basedir, pipeline):
    #TODO
    return ""

def _calc_inputsize(s_rate, t_span, c_no):
    return (c_no, s_rate * t_span)

def _get_dataloaders(traindir, evaldir, batch_size):
    trainloader = DataLoader(traindir, batch_size=batch_size, shuffle=True)
    evalloader = DataLoader(evaldir, batch_size=batch_size, shuffle=True)
    return trainloader, evalloader


def oneloop(device, model, input_size, datapath, basedir, pipeline, hyperparameters, trainelements, destdir):

    # We will start by initializing the model and data description
    model_description = {}
    model_description['name'] = model.__class__.__name__
    data_description = {}

    # Resetting cuda for every loop
    if device == 'cuda':
        torch.cuda.empty_cache()

    # Extracting the necessary elements
    metrics = trainelements.metrics
    history = trainelements.history
    criterion = trainelements.criterion
    earlystopping = trainelements.earlystopping

    # Generating the model and data summaries
    modelsummary = _get_modelsummary(model, input_size)
    datasummary = _get_datasummary(datapath, basedir, pipeline)


    # Making the dataset from its pipeline
    dataset = Dataset(datapath, basedir)
    dataset.set_pipeline(pipeline)
    datadir = os.path.join(destdir, 'data')
    traindir, evaldir, s_rate, t_span, c_no, data_id = dataset.save_all(datadir)
    if (input_size != _calc_inputsize(s_rate, t_span, c_no)):
        raise ValueError("Input Size Mismatch")

    # Adding the data description
    data_description['id'] = data_id
    data_description['pipeline'] = pipeline.__class__.__name__
    data_description['sampling_rate'] = s_rate
    data_description['time_span'] = t_span
    data_description['channel_no'] = c_no
    data_description['summary'] = datasummary

    # Adding the model description
    model_description['summary'] = modelsummary
    model_description['id'] = model.__class__.__name__ + '_' + data_id

    # Checking if the input size is correct
    train_loader, eval_loader = _get_dataloaders(traindir, evaldir, hyperparameters['batch_size'])
    optimizer = trainelements.optimizer(model.parameters(), lr=hyperparameters['lr'])



    # Finally training everything
    model_save_name = model_description['id'] + '.pt'
    model_save_path = os.path.join(destdir, 'models', model_save_name)
    respath = os.path.join(destdir, 'results.csv')
    train(model, train_loader, eval_loader, optimizer, criterion, hyperparameters['epochs'], history, metrics, device, model_save_path, earlystopping)
    update_csv(respath, model_description, data_description, history, hyperparameters)

    







    


