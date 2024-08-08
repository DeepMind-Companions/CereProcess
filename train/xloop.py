from torchsummary import summary
import os
import torch
from train.train import train, evaluate
from train.callbacks import History, def_metrics
from datasets.dataset import Dataset
from datasets.pytordataset import EEGDataset
from torch.utils.data import DataLoader
from train.misc import EarlyStopping, get_model_size
from train.store import update_csv

# def evaluate(model, val_loader, criterion, device, metrics, history):

def _get_modelsummary(model, input_size):
    model_summary = model.__str__()
    return model_summary

def _get_datasummary(datapath, basedir, pipeline):
    #TODO
    return ""

def _calc_inputsize(s_rate, t_span, c_no):
    return (c_no, s_rate * t_span)

def _get_dataloaders(traindir, evaldir, batch_size):
    traindataset = EEGDataset(traindir) 
    evaldataset = EEGDataset(evaldir)
    trainloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)
    evalloader = DataLoader(evaldataset, batch_size=batch_size, shuffle=True)
    return trainloader, evalloader


def _write_counter(counter, destpath, filename='counter.txt'):
    filename = os.path.join(destpath, filename)
    with open(filename, 'w') as file:
        file.write(str(counter))

def _read_counter(destpath, filename='counter.txt'):
    filename = os.path.join(destpath, filename)
    if not os.path.exists(filename):
        return 0  # Return a default value (e.g., 0) if the file does not exist
    
    with open(filename, 'r') as file:
        content = file.read()
        return int(content)  # or float(content) if you expect a float

def _increment_counter(destpath, filename='counter.txt'):
    filename = os.path.join(destpath, filename)
    counter = _read_counter(destpath)
    counter += 1
    _write_counter(counter, destpath)


def oneloop(device, model, input_size, datapath, basedir, pipeline, hyperparameters, trainelements, destdir, model_name = None):

    # We will start by initializing the model and data description
    model_description = {}
    if model_name is None:
        model_description['name'] = model.__class__.__name__
    else:
        model_description['name'] = model_name
    model_description["size"] = str(get_model_size(model))

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
    model_save_dir = os.path.join(destdir, 'models')
    os.makedirs(model_save_dir, exist_ok=True)
    currmodels = os.listdir(model_save_dir)
    model_save_name = "model_" + str(_read_counter(destdir))
    _increment_counter(destdir)

    model_description['id'] = model_save_name

    model_save_path = os.path.join(model_save_dir, model_save_name + '.pt')
    earlystopping.path = model_save_path
    train(model, train_loader, eval_loader, optimizer, criterion, hyperparameters['epochs'], history, metrics, device, model_save_path, earlystopping)
    update_csv(destdir, model_description, data_description, history, hyperparameters, model_save_name)

    







    


