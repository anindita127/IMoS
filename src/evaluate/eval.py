import numpy as np
import torch
import os
import sys
sys.path.append('.')
sys.path.append('..')
import math
import torch.nn.functional as F
from bps_torch.bps import bps_torch
from smplx import SMPLXLayer
from src.data_loader.dataloader import *
from src.evaluate.evaluate_statistical_metrics import *
from src.models.action_rec import LSTM_Action_Classifier
from src.models.model import *
from src.models.model_utils import batch_parms_6D2smplx_params
from src.tools.argUtils import argparseNloop
from src.tools.BookKeeper import *
from src.tools.meshviewer import Mesh
from src.tools.objectmodel import ObjectModel
from torch.utils.data import DataLoader
from tqdm import tqdm


base_path = ''
smplx_rhand_ver = np.load(os.path.join(base_path, '..', 'DATASETS', 'GRAB', 'tools', 'smplx_correspondence', 'rhand_smplx_ids.npy'))
smplx_lhand_ver = np.load(os.path.join(base_path, '..', 'DATASETS', 'GRAB', 'tools', 'smplx_correspondence', 'lhand_smplx_ids.npy'))
verts_ids = to_tensor(np.load(os.path.join('src', 'consts', 'verts_ids_0512.npy')), dtype=torch.long)
object_mesh_path = os.path.join('tools','object_meshes', 'contact_meshes')
subject_mesh_path = os.path.join('tools','subject_meshes')
used_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
right_hand_finger_joints = [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]
left_hand_finger_joints = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
upperbody_joints = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
right_arm_joints = [17, 19, 21]
arm_joints = [16, 17, 18, 19, 20, 21]
left_arm_joints = [16, 18, 20]
lowerbody_joints = [0, 1, 2, 4, 5, 7, 8, 10, 11]
rest_body_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
head_joint = [15, 22, 23, 24]
torso_joint = [0, 3, 6, 9, 12, 13, 14]
lowerbody_joints = [1, 2, 4, 5, 7, 8, 10, 11]

class Tester:
    def __init__(self,args, ds_name='test'):      
        torch.manual_seed(args.seed)
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.empty_cache()
        self.device = torch.device("cuda:%d" % args.cuda if torch.cuda.is_available() else "cpu")
        self.bps_torch = bps_torch(n_bps_points=512, random_seed=args.seed)
        self.dtype = torch.float32
        self.time = 15
        self.batch_size = 1

        '''Load the action classifier model'''
        args.load = os.path.join('save', 'pretrained_models', 'exp_121_model_CVAE_object_nojoint_lr_0-0005_batchsize_1_',
        'exp_121_model_CVAE_object_nojoint_lr_0-0005_batchsize_1_000500.p')
        args_subset0 = ['exp', 'model', 'lr', 'batch_size']
        self.book0 = BookKeeper(args, args_subset0)
        self.args0 = self.book0.args
        self.action_rec_model = LSTM_Action_Classifier().to(self.device).float()
        self.book0._load_model(self.action_rec_model, 'model_pose') 

        args_subset = ['exp', 'model', 'lr', 'batch_size', 'latentD', 'language_model', 'use_discriminator']
        '''Load the arms model'''
        args.load = os.path.join('save', 'pretrained_models', 'exp_80_model_Interaction_Prior_CVAE_lr_0-001_batchsize_64_latentD_32_languagemodel_clip_usediscriminator_False_',
        'exp_80_model_Interaction_Prior_CVAE_lr_0-001_batchsize_64_latentD_32_languagemodel_clip_usediscriminator_False_001600.p')
        assert os.path.exists(args.load), 'Arms model file not found' 
        self.book1 = BookKeeper(args, args_subset)
        self.args1 = self.book1.args
        self.arms_model = Interaction_Prior_CVAE(self.args1, self.device, self.time).to(self.device).float()
        print('Loading Arms Model')
        self.book1._load_model(self.arms_model, 'model_pose') 
        args.load = os.path.join('save', 'pretrained_models', 'exp_31_model_Interaction_Prior_Posterior_CVAE_clip_lr_0-0005_batchsize_64_latentD_100_languagemodel_clip_usediscriminator_False_',
        'exp_31_model_Interaction_Prior_Posterior_CVAE_clip_lr_0-0005_batchsize_64_latentD_100_languagemodel_clip_usediscriminator_False_001500.p')
        assert os.path.exists(args.load), 'Body model file not found' 
        self.book2 = BookKeeper(args, args_subset)
        self.args2 = self.book2.args
        self.body_model = Interaction_Prior_Posterior_CVAE_clip(self.args2, self.device, self.time).to(self.device).float()
        print('Loading Body Model')
        self.book2._load_model(self.body_model, 'model_pose') 
        print('Loading the data')
        self.load_data(args, ds_name)
        print('Data loading completed')
        self.config_optimizers()

    def config_optimizers(self):
        bs = 1
        self.bs = bs
        device = self.device
        dtype = self.dtype
        self.female_model = SMPLXLayer(
            model_path=os.path.join(args.model_path, 'smplx'),
            gender='female',
            num_pca_comps=45,
            flat_hand_mean=True,
            ).to(self.device)
        self.male_model = SMPLXLayer(
            model_path=os.path.join(args.model_path, 'smplx'),
            gender='male',
            num_pca_comps=45,
            flat_hand_mean=True,
            ).to(self.device)
        self.object_model = ObjectModel().to(self.device)
        

    def load_data(self,args, ds_name):
        if ds_name == 'test':
            ds_test = LoadData_GRAB(args, dataset_dir=args.dataset_dir, ds_name=ds_name)
            self.ds_test = DataLoader(ds_test, batch_size=1, num_workers=0, shuffle=False)
        elif ds_name == 'train':
            ds_test = LoadData_GRAB(args, dataset_dir=args.dataset_dir, ds_name=ds_name, load_on_ram=args.load_on_ram)
            self.ds_test = DataLoader(ds_test, batch_size=1, num_workers=0, shuffle=False)
        elif ds_name == 'val':
            ds_test = LoadData_GRAB(args, dataset_dir=args.dataset_dir, ds_name=ds_name, load_on_ram=args.load_on_ram)
            self.ds_test = DataLoader(ds_test, batch_size=1, num_workers=0, shuffle=False)
        
    def load_batch(self, batch, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        fullpose_with_fingers =  batch['fullpose'][:, :self.time, ...].to(self.device).float()
        transl =  batch['transl'][:, :self.time, ...].to(self.device).float().reshape(batch_size, self.time, -1)
        sbj_joints = batch['joints'][:, :self.time, ...].reshape(-1, self.time, 127, 3).to(self.device).float()
        fullpose_rot =  fullpose_with_fingers[:, :, used_joints, :].to(self.device).float()
        right_arm_rot =  fullpose_with_fingers[:, :, right_arm_joints, :].to(self.device).float().reshape(batch_size, self.time, -1)
        left_arm_rot =  fullpose_with_fingers[:, :, left_arm_joints, :].to(self.device).float().reshape(batch_size, self.time, -1)
        arms_rot = torch.cat((right_arm_rot, left_arm_rot), dim=-1)
        right_hand_finger_rot =  fullpose_with_fingers[:, :, right_hand_finger_joints, :].to(self.device).float().reshape(batch_size, self.time, -1)
        lefthand_finger_rot =  fullpose_with_fingers[:, :, left_hand_finger_joints, :].to(self.device).float().reshape(batch_size, self.time, -1)
        upperbody_rot =  fullpose_with_fingers[:, :, upperbody_joints, :].to(self.device).float().reshape(batch_size, self.time, -1)
        lowerbody_rot =  fullpose_with_fingers[:, :, lowerbody_joints, :].to(self.device).float().reshape(batch_size, self.time, -1)
        rest_body_rot =  fullpose_with_fingers[:, :, rest_body_joints, :].to(self.device).float().reshape(batch_size, self.time, -1)
        betas = batch['betas'].squeeze(1).to(self.device).float()
        rhand_contact = batch['rhand_contact'][:, :self.time].to(self.device).float()
        lhand_contact = batch['lhand_contact'][:, :self.time].to(self.device).float()
        rhand_contact_map = batch['rhand_contact_map'][:, :self.time].to(self.device).float()
        lhand_contact_map = batch['lhand_contact_map'][:, :self.time].to(self.device).float()
        right_hand_vert = batch['rhand_verts'][:, :self.time].to(self.device).float()
        left_hand_vert = batch['lhand_verts'][:, :self.time].to(self.device).float()
        verts = batch['verts'][:, :self.time].to(self.device).float()
        verts_obj = batch['verts_obj'][:, :self.time].to(self.device).float()
       
        obj_global_orient = batch['global_orient_obj'][:, :self.time].to(self.device).float().reshape(batch_size, self.time, -1)
        obj_transl = batch['transl_obj'][:, :self.time].to(self.device).float()
        data = {
            'joints': sbj_joints,
            'gender': batch['gender'] ,
            'sbj_v_template': batch['sbj_v_template'],
            'upperbody_rot': upperbody_rot,
            'right_arm_rot': right_arm_rot,
            'rest_body_rot': rest_body_rot,
            'left_arm_rot': left_arm_rot,
            'arms_rot': arms_rot,
            'righthand_finger_rot': right_hand_finger_rot,
            'lefthand_finger_rot': lefthand_finger_rot,
            'lowerbody_rot': lowerbody_rot,
            'fullpose': fullpose_rot,
            'fullpose_with_fingers': fullpose_with_fingers,
            'transl': transl,
            'betas': betas,
            'rhand_contact_map': rhand_contact_map,
            'lhand_contact_map': lhand_contact_map,
            'rhand_contact': rhand_contact,
            'lhand_contact': lhand_contact,
            'obj_transl': obj_transl,
            'obj_global_orient': obj_global_orient,
            'right_hand_verts': right_hand_vert,
            'left_hand_verts': left_hand_vert,
            'verts': verts,
            'verts_obj': verts_obj,
            'obj_vec': batch['obj_vec'].to(self.device),
            'sbj_id': batch['sbj_id'],
            'intent_vec': batch['intent_vec'].to(self.device),
            'sentence_vec': batch['sentence_vec'].to(self.device),
            'sentence': batch['sentence'] ,
            'intent_embedding': batch['intent_embedding'].to(self.device),
            'obj_embedding': batch['obj_embedding'].to(self.device),
            'obj_name': batch['obj_name'][0],
            'intent': batch['intent'][0],
        }
        return data

    def eval(self, ds_name='test'):
        self.arms_model.eval() 
        self.body_model.eval() 
        test_loader = self.ds_test
        test_tqdm = tqdm(test_loader, desc= ds_name + ' {:.10f}'.format(0), leave=False, ncols=100)        
        action_labels_gt = []
        JTS_gt = []
        JTS_sbj = torch.zeros((len(test_tqdm), 15, 22, 3)).to(device)
        y_pred = torch.zeros((len(test_tqdm), 29)).to(device)
        pred_activations = torch.zeros((len(test_tqdm), 128)).to(device)
        for count, batch in enumerate(test_tqdm):           
            T = self.time 
            data = self.load_batch(batch, batch_size=1)
            arms_rot_out = self.arms_model.sample(self, T, data)
            arms_rot_out = arms_rot_out.reshape(1, T, -1, 6)
            arms_rot_body_in = arms_rot_out

            '''Do the following rearrangement if you use the pre-trained model. There is a mismatch in the joint ordering in the pre-trained weights.'''
            '''Comment out the next 7 lines if you train your own model'''
            arms_rot_body_in = to_tensor(torch.zeros_like(arms_rot_out))
            arms_rot_body_in[..., 0, :] = arms_rot_out[..., 3, :]
            arms_rot_body_in[..., 1, :] = arms_rot_out[..., 0, :]
            arms_rot_body_in[..., 2, :] = arms_rot_out[..., 4, :]
            arms_rot_body_in[..., 3, :] = arms_rot_out[..., 1, :]
            arms_rot_body_in[..., 4, :] = arms_rot_out[..., 5, :]
            arms_rot_body_in[..., 5, :] = arms_rot_out[..., 2, :]
            
            body_rot_out = self.body_model.sample(self, T, data, arms_rot_body_in.reshape(1, T, -1))
            body_rot_out = body_rot_out.reshape(1, T, -1, 6)
            test_fullpose = data['fullpose_with_fingers'].clone()
            test_fullpose[:, :, arm_joints] = arms_rot_body_in
            test_fullpose[:, :, rest_body_joints] = body_rot_out
            sbj_p = batch_parms_6D2smplx_params(test_fullpose, data['transl'])   
            gender = data['gender']
            sbj_p['v_template'] = data['sbj_v_template'].to(self.device).float()
            if gender == 1:
                m_output = self.male_model(**sbj_p)
                joints_sbj = m_output.joints[:, :22]
            if gender == 2:
                f_output = self.female_model(**sbj_p)
                joints_sbj = f_output.joints[:, :22]
            action_labels_gt.append(data['intent_vec'])
            JTS_gt.append(data['joints'][..., :22, :])
            JTS_sbj[count] = joints_sbj
            
            y_pred[count], pred_activations[count] = self.action_rec_model.predict(joints_sbj)
            
        jt_gt = torch.cat(JTS_gt, dim=0)
        mean_ape_loss = mean_l2di_(torch.as_tensor(JTS_sbj), jt_gt).item()
        print("mean APE: ", mean_ape_loss)
        mean_ave_loss = torch.zeros((len(JTS_sbj)))
        for idx in range(len(JTS_sbj)):
            mean_ave_loss[idx] = calc_AVE(JTS_sbj[idx], jt_gt[idx]).item()
        
        print("mean AVE: ", torch.mean(mean_ave_loss))
        rec_accuracy = calculate_accuracy(JTS_sbj, 29, self.action_rec_model, action_labels_gt).item()
        print("Recognition Accuracy", rec_accuracy)
        fid = evaluate_fid(JTS_gt, JTS_sbj, 29, self.action_rec_model, action_labels_gt).item()
        print("FID score", fid)
        divers = calculate_diversity_(pred_activations, y_pred, 29).item()
        print("Diversity", divers)
        multimodality = calculate_multimodality_(pred_activations, y_pred, 29).item()
        print("Multimodality", multimodality)



if __name__ == '__main__':
    args = argparseNloop()
    ds_name = 'test'
    model_tester = Tester(args=args, ds_name=ds_name)
    print(" **Inititalization done**")
    model_tester.eval(ds_name)
