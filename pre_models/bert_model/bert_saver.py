
class RunningAverage():
    """A simple class that maintains the running average of a quantity
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """
    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)

def accuracy(y_true, y_pred):
    import numpy as np
    import torch

    batch_size = len(y_true)
    y_pred_labels = torch.max(y_pred, 1)[1].view(y_true.size())
    num_corr = (y_pred_labels.data==y_true.data).float().sum()
    return 100.0*num_corr/batch_size

def ner_accuracy(y_true, y_pred):
    from sklearn.metrics import accuracy_score
    #print(type(outputs), type(labels))
    #print(len(outputs), labels.shape)
    pred_labels = [i for ii in y_pred for i in ii]
    true_labels = [i.item() for ii in y_true for i in ii]
    #print(len(pred_labels), len(true_labels))
    #print(pred_labels[0], true_labels[0])
    return accuracy_score(true_labels, pred_labels)*100.0       

def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    import json
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)
        
def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    import os
    import shutil
    import torch
    
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))

def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    import os
    import shutil
    import torch
    
    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))
    # change map_location to cuda as wish
    checkpoint = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint 

def load_seq2seq_checkpoint(checkpoint, enc_model, dec_model, enc_optimizer=None, dec_optimizer=None):
    """
    !!! Warnning: This func only works for seq2seq model. DO NOT USE THIS FOR ANYTHING ELSE
    
    Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        enc_model: (torch.nn.Module) model for which the parameters are loaded
        enc_optimizer: (torch.optim) optional: resume optimizer from checkpoint
        dec_model: (torch.nn.Module) model for which the parameters are loaded
        dec_optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    import os
    import shutil
    import torch
    
    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))
    # change map_location to cuda as wish
    checkpoint = torch.load(checkpoint, map_location='cpu')
    
    enc_model.load_state_dict(checkpoint['encoder_state_dict'])
    dec_model.load_state_dict(checkpoint['decoder_state_dict'])

    if enc_optimizer:
        enc_optimizer.load_state_dict(checkpoint['encoder_optim_dict'])
    if dec_optimizer:
        dec_optimizer.load_state_dict(checkpoint['decoder_optim_dict'])

    return checkpoint 
