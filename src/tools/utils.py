import numpy as np
import torch
import logging
import math
import torch.nn.functional as F
from copy import copy
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
to_cpu = lambda tensor: tensor.detach().cpu().numpy()


def parse_npz(npz, allow_pickle=True):
    npz = np.load(npz, allow_pickle=allow_pickle)
    npz = {k: npz[k].tolist() for k in npz.files}
    return DotDict(npz)

def params2torch(params, dtype = torch.float32):
    return {k: torch.from_numpy(v).type(dtype).to(device) for k, v in params.items()}

def prepare_params(params, frame_mask, rel_trans = None, dtype = np.float32):
    n_params = {k: v[frame_mask].astype(dtype)  for k, v in params.items()}
    if rel_trans is not None:
        n_params['transl'] -= rel_trans
    return n_params

def torch2np(item, dtype=np.float32):
    out = {}
    for k, v in item.items():
        if v ==[] or v=={}:
            continue
        if isinstance(v, list):
            if isinstance(v[0],  str):
                out[k] = v
            else:
                if torch.is_tensor(v[0]):
                    v = [v[i].cpu() for i in range(len(v))]
                try:
                    out[k] = np.array(np.concatenate(v), dtype=dtype)
                except:
                    out[k] = np.array(np.array(v), dtype=dtype)
        elif isinstance(v, dict):
            out[k] = torch2np(v)
        else:
            if torch.is_tensor(v):
                v = v.cpu()
            out[k] = np.array(v, dtype=dtype) 
            
    return out

def DotDict(in_dict):
    out_dict = copy(in_dict)
    for k,v in out_dict.items():
       if isinstance(v,dict):
           out_dict[k] = DotDict(v)
    return dotdict(out_dict)

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def append2dict(source, data):
    for k in data.keys():
        if k in source.keys():
            if isinstance(data[k], list):
                source[k] += data[k]
            else:
                source[k].append(data[k])

            
def append2list(source, data):
    # d = {}
    for k in data.keys():
        leng = len(data[k])
        break
    for id in range(leng):
        d = {}
        for k in data.keys():
            if isinstance(data[k], list):
                if isinstance(data[k][0], str):
                    d[k] = data[k]
                elif isinstance(data[k][0], np.ndarray):
                    d[k] = data[k][id]
                
            elif isinstance(data[k], str):
                    d[k] = data[k]
            elif isinstance(data[k], np.ndarray):
                    d[k] = data[k]
        source.append(d)
           
        # source[k] += data[k].astype(np.float32)
        
        # source[k].append(data[k].astype(np.float32))

def np2torch(item, dtype=torch.float32):
    out = {}
    for k, v in item.items():
        if v ==[] :
            continue
        if isinstance(v, str):
           out[k] = v 
        elif isinstance(v, list):
            # if isinstance(v[0], str):
            #    out[k] = v
            try:
                out[k] = torch.from_numpy(np.concatenate(v)).to(dtype)
            except:
                out[k] = v # torch.from_numpy(np.array(v))
        elif isinstance(v, dict):
            out[k] = np2torch(v)
        else:
            out[k] = torch.from_numpy(v).to(dtype)
    return out

def to_tensor(array, dtype=torch.float32):
    if not torch.is_tensor(array):
        array = torch.tensor(array)
    return array.to(dtype).to(device)


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = np.array(array.todencse(), dtype=dtype)
    elif torch.is_tensor(array):
        array = array.detach().cpu().numpy()
    return array

def makepath(desired_path, isfile = False):
    '''
    if the path does not exist make it
    :param desired_path: can be path to a file or a folder name
    :return:
    '''
    import os
    if isfile:
        if not os.path.exists(os.path.dirname(desired_path)):os.makedirs(os.path.dirname(desired_path))
    else:
        if not os.path.exists(desired_path): os.makedirs(desired_path)
    return desired_path