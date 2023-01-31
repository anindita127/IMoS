import os
import sys
sys.path.append('.')
sys.path.append('..')
import glob
import joblib
import numpy as np
import torch
import smplx as smplx
from src.tools.argUtils import argparseNloop
from src.tools.utils import to_tensor
from torch.utils import data
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LoadData_GRAB(data.Dataset):
    def __init__(self,
                 args, 
                 dataset_dir,
                 ds_name='train',
                 dtype=torch.float32,
                 load_on_ram = False):

        super().__init__()
        current_dirpath = os.path.join(dataset_dir, ds_name)
        
        self.all_seqs = glob.glob(os.path.join(current_dirpath, '*.npz'), recursive = True) 
        self.args = args
        self.data_dict = {}
        total_num_seqs = len(self.all_seqs)
        skip = args.skip_train if ds_name.lower() == 'train' else (args.skip_val if ds_name.lower() == 'val' else 1)
        for data_count, data_idx in enumerate(tqdm(range(0, total_num_seqs, skip))):
            loaded = self.load(self.all_seqs[data_idx])
            self.data_dict[data_count] = loaded
        self.num_samples = data_count

       
    def load(self, dataset):
        loaded = torch.load(dataset)
        return loaded

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        items = self.data_dict[idx]
        T = items['fullpose'].shape[0]
        if T < self.args.max_frames:
            buff = self.args.max_frames - T
            fullpose = to_tensor(torch.zeros(buff, 55, 6))
            global_orient_obj = to_tensor(torch.zeros(buff, 1, 6))
            transl =  np.zeros((buff, 3))
            transl_obj =  np.zeros((buff, 3))
            verts =  np.zeros((buff, 400, 3))
            verts_obj =  np.zeros((buff, 512, 3))
            joints =  np.zeros((buff, 127, 3))
            rhand_verts =  np.zeros((buff, 778, 3))
            lhand_verts =  np.zeros((buff, 778, 3))
            verts2obj =  np.zeros((buff, 400, 3))
            bps_obj_glob =  np.zeros((buff, 1024, 3))
            rhand_contact =  np.zeros(buff)
            lhand_contact =  np.zeros(buff)
            rhand_contact_map =  np.zeros((buff, 778))
            lhand_contact_map =  np.zeros((buff, 778))
            items['fullpose'] = torch.cat((to_tensor(items['fullpose']), fullpose))
            items['global_orient_obj'] = torch.cat((to_tensor(items['global_orient_obj']), global_orient_obj))
            items['transl'] =  np.concatenate(( items['transl'], transl), axis =0)
            items['transl_obj'] =  np.concatenate(( items['transl_obj'], transl_obj), axis=0)
            items['verts'] =  np.concatenate(( items['verts'], verts), axis=0)
            items['verts_obj'] =  np.concatenate(( items['verts_obj'], verts_obj), axis=0)
            items['verts2obj'] =  np.concatenate(( items['verts2obj'], verts2obj), axis=0)
            items['joints'] =  np.concatenate(( items['joints'], joints), axis=0)
            items['bps_obj_glob'] =  np.concatenate(( items['bps_obj_glob'], bps_obj_glob), axis=0)
            items['rhand_verts'] =  np.concatenate(( items['rhand_verts'], rhand_verts), axis=0)
            items['lhand_verts'] =  np.concatenate(( items['lhand_verts'], lhand_verts), axis=0)
            items['rhand_contact'] =  np.concatenate(( items['rhand_contact'], rhand_contact), axis=0)
            items['lhand_contact'] =  np.concatenate(( items['lhand_contact'], lhand_contact), axis=0)
            if 'rhand_contact_map' in items.keys():
                items['rhand_contact_map'] =  np.concatenate(( items['rhand_contact_map'], rhand_contact_map), axis=0)
                items['lhand_contact_map'] =  np.concatenate(( items['lhand_contact_map'], lhand_contact_map), axis=0)

        if isinstance(items['gender'], str) and items['gender'].lower() == 'male':
            items['gender'] = 1
        elif isinstance(items['gender'], str) and items['gender'].lower() == 'female':
            items['gender'] = 2
        return items

def collate_fn(data):
    return data


if __name__=='__main__':
    args = argparseNloop()
    data_path = args.dataset_dir 
    ds = LoadData_GRAB(args, data_path, ds_name='val')
    bs = 16
    T = 20
    dataloader = data.DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0, drop_last=True)
    for count, batch in enumerate(dataloader):
        tmp1 = torch.linalg.norm(batch['joints'][...,:22, :], dim=-1)
        tmp2 = tmp1[:, 1:] - tmp1[:, :-1]
        t_m, t_ind = torch.sort(torch.mean(torch.mean(tmp2, dim=1), dim=0), descending=True)
        print(count)
    print('end')