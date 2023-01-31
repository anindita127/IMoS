import clip 
import glob
import json
import numpy as np
import os
import sys
sys.path.append('.')
sys.path.append('..')
import torch
from bps_torch.bps import bps_torch
from smplx import SMPLXLayer
from src.tools.argUtils import argparseNloop
from src.tools.meshviewer import Mesh
from src.models.model_utils import batch_parms_6D2smplx_params
from src.tools.objectmodel import ObjectModel
from src.tools.transformations import aa2d6, to_tensor
from src.tools.utils import makepath, to_cpu, parse_npz
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_path = ''
smplx_rhand_ver = np.load(os.path.join(base_path, '..', 'DATASETS', 'GRAB', 'tools', 'smplx_correspondence', 'rhand_smplx_ids.npy'))
smplx_lhand_ver = np.load(os.path.join(base_path, '..', 'DATASETS', 'GRAB', 'tools', 'smplx_correspondence', 'lhand_smplx_ids.npy'))
verts_ids = to_tensor(np.load(os.path.join('src', 'consts', 'verts_ids_0512.npy')), dtype=torch.long)

expr_dir = os.path.join('human_body_prior','vposer_v2_05')

object_list = ['cylinderlarge', 'mug', 'elephant', 'hand', 'cubelarge', 'stanfordbunny', 'airplane', 'alarmclock', 'banana', 'body', 'bowl', 'cubesmall', 'cup', 'doorknob', 'cubemedium',
             'eyeglasses', 'flashlight', 'flute', 'gamecontroller', 'hammer', 'headphones', 'knife', 'lightbulb', 'mouse', 'phone', 'piggybank', 'pyramidlarge', 'pyramidsmall', 'pyramidmedium',
             'duck', 'scissors', 'spherelarge', 'spheresmall', 'stamp', 'stapler', 'table', 'teapot', 'toruslarge', 'torussmall', 'train', 'watch', 'cylindersmall', 'waterbottle', 'torusmedium',
             'cylindermedium', 'spheremedium', 'wristwatch', 'wineglass', 'camera', 'binoculars', 'fryingpan', 'toothpaste', 'apple', 'toothbrush']
intent_list = ['offhand','pass', 'lift', 'drink', 'brush', 'eat', 'peel', 'takepicture', 'see', 'wear', 'play', 'clean', 'browse', 'inspect', 'pour', 'use', 
                'switchON', 'cook', 'toast', 'staple', 'squeeze', 'set', 'open', 'chop', 'screw', 'call', 'shake', 'fly', 'stamp']

mapping_intent = {}
mapping_object = {}
for x in range(len(intent_list)):
  mapping_intent[intent_list[x]] = x   

for x in range(len(object_list)):
  mapping_object[object_list[x]] = x   

def one_hot_vectors(word, word_list, mapping):
    arr = list(np.zeros(len(word_list), dtype = int))
    arr[mapping[word]] = 1
    return np.array(arr)

plural_dict = {
'offhand': 'offhands',
'pass': 'passes',
'lift': 'lifts', 
'drink': 'drinks from', 
'brush': 'brushes with',
'eat': 'eats', 
'peel': 'peels', 
'takepicture': 'takes picture with', 
'see': 'sees in', 
'wear': 'wears', 
'play': 'plays', 
'clean': 'cleans', 
'browse': 'browses on', 
'inspect': 'inspects', 
'pour': 'pours from', 
'use': 'uses', 
'switchON': 'switches on', 
'cook': 'cooks on', 
'toast': 'toasts with ', 
'staple': 'staples with', 
'squeeze': 'squeezes', 
'set': 'sets', 
'open': 'opens', 
'chop': 'chops with', 
'screw': 'screws', 
'call': 'calls on', 
'shake': 'shakes', 
'fly': 'flies',
'stamp': 'stamps with'    
}

class GRABDataSet(object):
    def __init__(self, args, cfg):
        self.cfg = cfg
        self.args = args
        self.data_path = args.data_path
        self.out_path = args.dataset_dir
        args.max_frames = 20 #for our experiments
        self.max_frames = args.max_frames
        self.framerate = args.framerate
        language_model = args.language_model
        makepath(self.out_path)
        print('Starting data preprocessing !')
        self.intent = cfg['intent']
        print('intent: ',self.intent)
        # self.splits = cfg['splits']
        self.all_seqs = glob.glob(self.data_path + '/**/*.npz', recursive = True)
        self.selected_seqs = []
        self.obj_based_seqs = {}
        self.sbj_based_seqs = {}
        self.split_seqs = {
            'test': [],
            'train': [],
            'val': [],
            }
        self.process_sequences()
        print('Number of subjects in each data split : train: %d , test: %d , val: %d'
                         % (len(self.cfg['split_train']), len(self.cfg['split_test']), len(self.cfg['split_val'])))
        
        self.bps_torch = bps_torch(n_bps_points=512, random_seed=args.seed)
        self.bps = torch.load(os.path.join('src', 'consts', 'bps.pt'))
        self.data_preprocessing(args, cfg, language_model)

    def process_sequences(self):
        for sequence in self.all_seqs:
            sequence_fname = sequence.split('/')[-1]
            subject_id = sequence.split('/')[-2]
            
            object_name = sequence_fname.split('_')[0]
            intent_type =  sequence_fname.split('_')[1]
            # filter data based on the motion intent
            if self.intent == 'all':
                pass
            elif intent_type not in self.intent:
                continue 
            
            if object_name not in self.obj_based_seqs:
                self.obj_based_seqs[object_name] = [sequence]
            else:
                self.obj_based_seqs[object_name].append(sequence)
            # group motion sequences based on subjects
            if subject_id not in self.sbj_based_seqs:
                self.sbj_based_seqs[subject_id] = [sequence]
            else:
                self.sbj_based_seqs[subject_id].append(sequence)
            # split train, val, and test sequences
            self.selected_seqs.append(sequence)
            if subject_id in self.cfg['split_test']:
                self.split_seqs['test'].append(sequence)
            elif subject_id in self.cfg['split_val']:
                self.split_seqs['val'].append(sequence)
            elif subject_id in self.cfg['split_train']:
                self.split_seqs['train'].append(sequence)
                
    def data_preprocessing(self, args, cfg, language_model = 'clip'):
        self.subject_mesh = {}
        self.obj_info = {}
        self.sbj_info = {}
        for split in self.split_seqs.keys():
            print('Processing data for %s split.' % (split))
            count_seq = 0
            for sequence in tqdm(self.split_seqs[split]):
                seq_data = parse_npz(sequence)
                if seq_data.motion_intent == 'lift':
                    continue
                obj_name = seq_data.obj_name
                sbj_id   = seq_data.sbj_id
                gender   = seq_data.gender
                GRAB_data = {
                    'intent': seq_data.motion_intent,
                    'seq_name': sequence,
                    'obj_name': obj_name,
                    'sbj_id': sbj_id,
                    'gender': gender,
                    
                    }
                GRAB_data['sentence'] = ' The person '+ plural_dict[seq_data.motion_intent] + ' the ' +   GRAB_data['obj_name'] +'.'
                model_clip, preprocess_clip = clip.load("ViT-B/32", device=device)
                GRAB_data['sentence_vec'] = model_clip.encode_text(clip.tokenize([GRAB_data['sentence']]).to(device)).cpu().detach().numpy()
                GRAB_data['intent_embedding'] = model_clip.encode_text(clip.tokenize([GRAB_data['intent']]).to(device)).cpu().detach().numpy()
                GRAB_data['obj_embedding'] = model_clip.encode_text(clip.tokenize([GRAB_data['obj_name']]).to(device)).cpu().detach().numpy()
                GRAB_data['intent_vec'] = one_hot_vectors(seq_data.motion_intent, intent_list, mapping_intent)
                GRAB_data['obj_vec'] = one_hot_vectors(GRAB_data['obj_name'], object_list, mapping_object)
                frame_mask, input_frame_idx = self.filter_contact_frames(seq_data)
                T = frame_mask.sum()
               
                if T < 1:
                    continue # if no frame is selected continue to the next sequence
                bs = T 
                sbj_mesh_path = os.path.join(args.data_path, seq_data.body.vtemp)
                GRAB_data['betas'] = np.load(file=sbj_mesh_path.replace('.ply','_betas.npy'))
                sbj_vtemp = self.load_sbj_verts(sbj_id, sbj_mesh_path, gender)
                GRAB_data['sbj_v_template'] = sbj_vtemp
                obj_mesh_path =  os.path.join(args.data_path, seq_data.object.object_mesh)
                obj_info = self.load_obj_verts(obj_name, obj_mesh_path, 512)

                with torch.no_grad():
                    sbj_m = SMPLXLayer(
                            model_path=os.path.join(args.model_path, 'smplx'),
                            gender=gender,
                            num_pca_comps=45,
                            flat_hand_mean=True).to(device)

                    obj_m = ObjectModel(v_template=obj_info['verts_sample'],
                                        batch_size=bs).to(device)
                sbj_params = seq_data.body.params
                obj_params = seq_data.object.params
                sbj_params['fullpose'] = sbj_params['fullpose'][input_frame_idx]
                sbj_params['body_pose'] = sbj_params['body_pose'][input_frame_idx]
                sbj_params['right_hand_pose'] = sbj_params['right_hand_pose'][input_frame_idx]
                sbj_params['left_hand_pose'] = sbj_params['left_hand_pose'][input_frame_idx]
                sbj_params['transl'] = sbj_params['transl'][input_frame_idx]
                sbj_params['global_orient'] = sbj_params['global_orient'][input_frame_idx]
                obj_params['global_orient'] = obj_params['global_orient'][input_frame_idx]
                obj_params['transl'] = obj_params['transl'][input_frame_idx]
                new_shape = sbj_params['transl'].shape[0]

                if cfg['shift_to_origin']:
                    '''translate pose to origin'''
                    init_trans_x = sbj_params['transl'][0,0]
                    init_trans_y = sbj_params['transl'][0,1]
                    init_trans_z = sbj_params['transl'][0,2]
                    sbj_params['transl'][:, 0] = sbj_params['transl'][:, 0] - init_trans_x
                    sbj_params['transl'][:, 1] = sbj_params['transl'][:, 1] - init_trans_y
                    sbj_params['transl'][:, 2] = sbj_params['transl'][:, 2] - init_trans_z 
                   
                    obj_params['transl'][:, 0] = obj_params['transl'][:, 0] - init_trans_x
                    obj_params['transl'][:, 1] = obj_params['transl'][:, 1] - init_trans_y
                    obj_params['transl'][:, 2] = obj_params['transl'][:, 2] - init_trans_z     
                if cfg['shift_obj_to_origin']:
                    '''translate obj to origin'''
                    init_trans_x = obj_params['transl'][0,0]
                    init_trans_y = obj_params['transl'][0,1]
                    init_trans_z = obj_params['transl'][0,2]
                    obj_params['transl'][:, 0] = obj_params['transl'][:, 0] - init_trans_x
                    obj_params['transl'][:, 1] = obj_params['transl'][:, 1] - init_trans_y
                    obj_params['transl'][:, 2] = obj_params['transl'][:, 2] - init_trans_z        
                    sbj_params['transl'][:, 0] = sbj_params['transl'][:, 0] - init_trans_x
                    sbj_params['transl'][:, 1] = sbj_params['transl'][:, 1] - init_trans_y
                    sbj_params['transl'][:, 2] = sbj_params['transl'][:, 2] - init_trans_z
                
                '''Convert axis angle to 6D representation'''
                GRAB_data['fullpose'] = aa2d6(sbj_params['fullpose'].reshape((new_shape, -1, 3)))
                GRAB_data['global_orient_obj'] = aa2d6(obj_params['global_orient'].reshape((new_shape, -1, 3)))
                GRAB_data['transl'] = sbj_params['transl']
                GRAB_data['transl_obj'] = obj_params['transl'] 
                sbj_p = batch_parms_6D2smplx_params(GRAB_data['fullpose'].unsqueeze(0), to_tensor(GRAB_data['transl']).unsqueeze(0))
                sbj_output = sbj_m(**sbj_p)
                verts_sbj = sbj_output.vertices
                obj_p = {
                    'transl': to_tensor(obj_params['transl']),
                    'global_orient': to_tensor(obj_params['global_orient']),
                }
                obj_out = obj_m(**obj_p)
                verts_obj = obj_out.vertices

                GRAB_data['verts']=to_cpu(verts_sbj[:, verts_ids])
                GRAB_data['joints']=to_cpu(sbj_output.joints)
                GRAB_data['rhand_verts']=to_cpu(verts_sbj[:, smplx_rhand_ver])
                GRAB_data['lhand_verts']=to_cpu(verts_sbj[:, smplx_lhand_ver])
                GRAB_data['verts_obj']=to_cpu(verts_obj)
                

                verts2obj = self.bps_torch.encode(x=verts_obj,
                                                    feature_type=['deltas'],
                                                    custom_basis=verts_sbj[:, verts_ids])['deltas']

                GRAB_data['verts2obj']=to_cpu(verts2obj)

                obj_bps = self.bps['obj'].to(device) + obj_p['transl'].reshape(T, 1, 3)

                bps_obj = self.bps_torch.encode(x=verts_obj,
                                                    feature_type=['deltas'],
                                                    custom_basis=obj_bps)['deltas']

                GRAB_data['bps_obj_glob']=to_cpu(bps_obj)
                GRAB_data['rhand_contact'] = np.zeros(T)
                GRAB_data['lhand_contact'] = np.zeros(T)
                GRAB_data['rhand_contact_map'] = np.zeros((T, 778))
                GRAB_data['lhand_contact_map'] = np.zeros((T, 778))
                for c, ind in enumerate(input_frame_idx):
                    GRAB_data['rhand_contact'][c] = (seq_data['contact']['body'][ind, smplx_rhand_ver]>0).any(axis=0) * 1
                    GRAB_data['lhand_contact'][c] = (seq_data['contact']['body'][ind, smplx_lhand_ver]>0).any(axis=0) * 1
                    GRAB_data['rhand_contact_map'][c] = seq_data['contact']['body'][ind, smplx_rhand_ver]
                    GRAB_data['lhand_contact_map'][c] = seq_data['contact']['body'][ind, smplx_lhand_ver]
                self.save_list(GRAB_data, split)
                count_seq += 1 
            print('Processing for', split, 'split finished with', count_seq, 'sequences')
       
    def save_list(self, data, split):
        d = {}
        for k in data.keys():
            if isinstance(data[k], list):
                if isinstance(data[k][0], str):
                    d[k] = data[k]
                elif isinstance(data[k][0], torch.Tensor) or isinstance(data[k][0], np.ndarray) or isinstance(data[k][0], np.int64) or isinstance(data[k][0], np.float64):
                    d[k] = data[k][id]
            elif isinstance(data[k], str):
                    d[k] = data[k]
            elif isinstance(data[k], torch.Tensor) or isinstance(data[k], np.ndarray) or isinstance(data[k], np.float64) or isinstance(data[k], np.int64):
                    d[k] = data[k]
        seq_name = data['seq_name'].split('/')[-2] + '_' + d['seq_name'].split('/')[-1][:-4] + '.npz'
        outfname = makepath(os.path.join(self.out_path, split, seq_name), isfile=True)
        d['seq_name'] = str(outfname)
        torch.save(d, outfname)         
    
    def filter_contact_frames(self,seq_data):
        pickup_frame_idx = np.array([int(np.nonzero(np.sum(seq_data['contact']['object'], axis=1))[0][0])])[0]
        putdown_frame_idx = np.array([int(np.nonzero(np.sum(seq_data['contact']['object'], axis=1))[0][-1])])[0]
        T = putdown_frame_idx - pickup_frame_idx
        frames_per_pose_input = T // args.max_frames 
        '''while taking standard framerate of 25 instead of variable framerate to downsample to key frames uncomment the next line ''' 
        input_frame_idx = np.arange(pickup_frame_idx , putdown_frame_idx, frames_per_pose_input)[:args.max_frames]
        masking = np.full(seq_data['contact']['body'].shape[0], False)
        masking[input_frame_idx] = True
        return masking, input_frame_idx
    
    def load_obj_verts(self, obj_name, mesh_path, n_verts_sample=512):
        if obj_name not in self.obj_info:
            np.random.seed(self.args.seed)
            obj_mesh = Mesh(filename=mesh_path)
            verts_obj = np.array(obj_mesh.vertices)
            faces_obj = np.array(obj_mesh.faces)

            if verts_obj.shape[0] > n_verts_sample:
                verts_sample_id = np.random.choice(verts_obj.shape[0], n_verts_sample, replace=False)
            else:
                verts_sample_id = np.arange(verts_obj.shape[0])

            verts_sampled = verts_obj[verts_sample_id]
            self.obj_info[obj_name] = {'verts': verts_obj,
                                       'faces': faces_obj,
                                       'verts_sample_id': verts_sample_id,
                                       'verts_sample': verts_sampled,
                                       'obj_mesh_file': mesh_path}
        return self.obj_info[obj_name]

    def load_sbj_verts(self, sbj_id, mesh_path, gender):
        betas_path = mesh_path.replace('.ply', '_betas.npy')
        if sbj_id in self.sbj_info:
            sbj_vtemp = self.sbj_info[sbj_id]['vtemp']
        else:
            sbj_vtemp = np.array(Mesh(filename=mesh_path).vertices)
            sbj_betas = np.load(betas_path)
            self.sbj_info[sbj_id] = {'vtemp': sbj_vtemp,
                                     'gender': gender,
                                     'betas': sbj_betas}
        return sbj_vtemp



if __name__ == '__main__':
    args = argparseNloop()
    cfg = {
        'intent': 'all',
        'only_contact': True, # if True, returns only frames with contact
        'save_body_verts': False, # if True, will compute and save the body vertices
        'save_lhand_verts': False, # if True, will compute and save the lhand vertices
        'save_rhand_verts': False, # if True, will compute and save the rhand vertices
        'save_object_verts': True,
        'save_contact': True, # if True, will add the contact info to the saved data
        'shift_to_origin': True,
        'shift_obj_to_origin': False,
        'split_array': True,
        'framerate': args.framerate,
        'max_frames': args.max_frames,
        'window_size': 15,
        'split_test': ['s10'],
        'split_val': ['s1', ],
        'split_train': ['s2', 's3', 's4', 's5', 's6', 's7', 's8', 's9'],
    }
    outfname = makepath(os.path.join(args.dataset_dir, 'preprocessing_logs.txt'), isfile=True)
    with open(outfname, 'w') as file: 
         file.write(json.dumps(cfg))
    GRABDataSet(args, cfg)