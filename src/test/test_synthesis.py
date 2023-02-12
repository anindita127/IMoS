import glob
import os
import shutil
import sys
sys.path.append('.')
sys.path.append('..')
import json
import numpy as np
import torch
import torch.nn.functional as F
from bps_torch.bps import bps_torch
from cmath import nan
from datetime import datetime
from omegaconf import OmegaConf
from scipy.optimize import least_squares
from smplx import SMPLXLayer
from smplx.lbs import batch_rodrigues
from src.data_loader.dataloader import *
from src.models.cvae_gnet import gnet_model
from src.models.model import *
from src.models.model_utils import batch_parms_6D2smplx_params
from src.tools.argUtils import argparseNloop
from src.tools.BookKeeper import *
from src.tools.meshviewer import Mesh
from src.tools.transformations import aa2d6, d62aa, rotmat2d6
from src.tools.utils import makepath, to_tensor
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
object_mesh_path = os.path.join('tools','object_meshes', 'new_contact_meshes')
subject_mesh_path = os.path.join('tools','subject_meshes')
vposer_expr_dir = os.path.join('human_body_prior','vposer_v2_05')
used_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
right_hand_finger_joints = [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]
left_hand_finger_joints = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
upperbody_joints = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
right_arm_joints = [17, 19, 21]
arm_joints = [16, 17, 18, 19, 20, 21]
left_arm_joints = [16, 18, 20]
lowerbody_joints = [0, 1, 2, 4, 5, 7, 8, 10, 11]
rest_body_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
scale = 10

class Tester:
    def __init__(self, args, ds_name='test'):      
        torch.manual_seed(args.seed)
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.empty_cache()
        self.device = torch.device("cuda:%d" % args.cuda if torch.cuda.is_available() else "cpu")
        self.bps_torch = bps_torch(n_bps_points=512, random_seed=args.seed)
        self.dtype = torch.float32
        cfg_path =  os.path.join('src', 'configs', 'GNet_orig.yaml')
        cfg = OmegaConf.load(cfg_path)
        cfg.best_model = os.path.join('src', 'consts', 'GNet_model.pt')
        self.cfg = cfg
        self.network = gnet_model(**cfg.network.gnet_model).to(self.device)
        self._get_network().load_state_dict(torch.load(cfg.best_model, map_location=self.device), strict=False)
        self.time = 15 
        self.batch_size = 1
        args_subset = ['exp', 'model', 'lr', 'batch_size', 'latentD', 'language_model', 'use_discriminator']

        '''Load the arms model'''
        args.load = os.path.join('save', 'pretrained_models', 'exp_80_model_Interaction_Prior_CVAE_lr_0-001_batchsize_64_latentD_32_languagemodel_clip_usediscriminator_False_',
        'exp_80_model_Interaction_Prior_CVAE_lr_0-001_batchsize_64_latentD_32_languagemodel_clip_usediscriminator_False_001600.p')
        assert os.path.exists(args.load), 'Arms model file not found' 
        self.book1 = BookKeeper(args, args_subset)
        self.args1 = self.book1.args
        self.arms_model = Interaction_Prior_CVAE(self.args1, self.device, self.time).to(self.device).float()
        print('Arms Model Created')
        self.book1._load_model(self.arms_model, 'model_pose') 
        
        args.load = os.path.join('save', 'pretrained_models', 'exp_31_model_Interaction_Prior_Posterior_CVAE_clip_lr_0-0005_batchsize_64_latentD_100_languagemodel_clip_usediscriminator_False_',
        'exp_31_model_Interaction_Prior_Posterior_CVAE_clip_lr_0-0005_batchsize_64_latentD_100_languagemodel_clip_usediscriminator_False_001500.p')
        assert os.path.exists(args.load), 'Body model file not found' 
        self.book2 = BookKeeper(args, args_subset)
        self.args2 = self.book2.args
        self.body_model = Interaction_Prior_Posterior_CVAE_clip(self.args2, self.device, self.time).to(self.device).float()
        print('Body Model Created')
        self.book2._load_model(self.body_model, 'model_pose') 

        self.load_data(args, ds_name)
        self.config_optimizers()

    def _get_network(self):
        return self.network.module if isinstance(self.network, torch.nn.DataParallel) else self.network

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
        
        self.opt_params = {
            'left_hand_pose'    : torch.randn(bs, 15*6, device=device, dtype=dtype, requires_grad=True),
            'right_hand_pose'   : torch.randn(bs, 15*6, device=device, dtype=dtype, requires_grad=True),
        }
        self.lr = self.cfg.get('smplx_opt_lr', 5e-3)
        
        self.optimizer = optim.Adam([self.opt_params[k] for k in ['right_hand_pose', 'left_hand_pose']], lr=self.lr)
        self.LossL1 = nn.L1Loss(reduction='mean')
        self.LossL2 = nn.MSELoss(reduction='mean')
        self.opt_num_iters = 160

    def init_params(self, finger_pose):
        for k in self.opt_params.keys():
            self.opt_params[k].data = torch.repeat_interleave(finger_pose[k], self.bs, dim=0)
  
    def gnet_infer(self, x):   #This is taken from https://github.com/otaheri/GOAL/blob/main/test/GOAL.py
        ##############################################
        bs = x['transl'].shape[0]
        dec_x = {}
        dec_x['betas'] = x['betas']
        dec_x['transl_obj'] = x['transl_obj']
        dec_x['bps_obj'] = x['bps_obj_glob'].norm(dim=-1)
        #####################################################
        z_enc = torch.distributions.normal.Normal(
            loc=torch.zeros([1, self.cfg.network.gnet_model.latentD], requires_grad=False).to(self.device).type(self.dtype),
            scale=torch.ones([1, self.cfg.network.gnet_model.latentD], requires_grad=False).to(self.device).type(self.dtype))
        z_enc_s = z_enc.rsample()
        dec_x['z'] = z_enc_s
        dec_x = torch.cat([v.reshape(bs, -1).to(self.device).type(self.dtype) for v in dec_x.values()], dim=1)
        net_output = self.network.decode(dec_x)
        pose, trans = net_output['pose'], net_output['trans']
        sbj_p = parms_6D2smplx_params(pose, trans)
        return sbj_p['right_hand_pose'], sbj_p['left_hand_pose']
    

    def opt_model(self, p, v_template, hand_vert, D):
        rotmat_obj = to_tensor(p[:3])
        translate_obj = to_tensor(p[3:6])
        transl_obj1 = translate_obj.reshape(-1, 1, 3)
        global_orient_obj = batch_rodrigues(rotmat_obj.view(-1, 3)).reshape([1, 3, 3])
        if v_template.ndim < 3:
            v_template = v_template.reshape(1, -1, 3)
        verts_obj = torch.matmul(to_tensor(v_template), global_orient_obj) + transl_obj1
        verts_obj = verts_obj.squeeze(0)
        obj_vertex = torch.repeat_interleave(verts_obj, len(hand_vert), dim=0)
        hand_vertex = to_tensor(hand_vert).repeat(verts_obj.shape[0], 1)
        diff = obj_vertex - hand_vertex
        d = torch.norm(diff.squeeze(0), dim=-1).cpu().detach().numpy()
        return d - D
        
 
    def calc_loss_fingers(self, smplx_model, sbj_p, obj_vertices_opt, rhand_flag): 
        sbj_opt = copy.deepcopy(sbj_p)
        sbj_opt['right_hand_pose'] = d62rotmat(self.opt_params['right_hand_pose']).unsqueeze(0)
        sbj_opt['left_hand_pose'] = d62rotmat(self.opt_params['left_hand_pose']).unsqueeze(0)
        sbj_verts = smplx_model(**sbj_p).vertices  
        right_hand_vertices = sbj_verts[:, smplx_rhand_ver]
        left_hand_vertices = sbj_verts[:, smplx_lhand_ver]
        sbj_verts_opt = smplx_model(**sbj_opt).vertices         
        right_hand_vertices_opt = sbj_verts_opt[:, smplx_rhand_ver]
        left_hand_vertices_opt = sbj_verts_opt[:, smplx_lhand_ver]
        losses = 0
        contact_threshold = 0.01
        init_obj_vertex1 = torch.repeat_interleave(obj_vertices_opt, right_hand_vertices.shape[1], dim=1)
        if rhand_flag: 
            rh2obj_opt = self.bps_torch.encode(x=obj_vertices_opt,
                                       feature_type=['deltas'],
                                       custom_basis=right_hand_vertices_opt)['deltas']
            rh2obj_pred = self.bps_torch.encode(x=obj_vertices_opt,
                                       feature_type=['deltas'],
                                       custom_basis=right_hand_vertices)['deltas'].detach()
            rh2obj_w_pred = torch.exp(-2 * rh2obj_pred.norm(dim=-1, keepdim=True))
            right_hand_vertices_opt1 = right_hand_vertices_opt.repeat(1, obj_vertices_opt.shape[1], 1)
            diff = init_obj_vertex1 - right_hand_vertices_opt1
            D_right = torch.norm(diff, dim=-1).float()
            D_right = D_right[:, ::300]
            r_flag = D_right < contact_threshold
            losses += 1*self.LossL1(rh2obj_w_pred * rh2obj_opt, rh2obj_w_pred * rh2obj_pred)
            losses += 0.005*torch.mean(D_right[r_flag] - 0.0)
            losses += 0.005*self.LossL1(right_hand_vertices_opt, right_hand_vertices)
        else: 
            lh2obj_opt = self.bps_torch.encode(x=obj_vertices_opt,
                                       feature_type=['deltas'],
                                       custom_basis=left_hand_vertices_opt)['deltas']
            lh2obj_pred = self.bps_torch.encode(x=obj_vertices_opt,
                                       feature_type=['deltas'],
                                       custom_basis=left_hand_vertices)['deltas'].detach()
            lh2obj_w_pred = torch.exp(-2 * lh2obj_pred.norm(dim=-1, keepdim=True))
            left_hand_vertices_opt1 = left_hand_vertices_opt.repeat(1, obj_vertices_opt.shape[1], 1)
            diff_left = init_obj_vertex1 - left_hand_vertices_opt1
            D_left = torch.norm(diff_left, dim=-1).float()
            D_left = D_left[:, ::300]
            l_flag = D_left < contact_threshold
            losses += 1*self.LossL1(lh2obj_w_pred * lh2obj_opt, lh2obj_w_pred * lh2obj_pred)
            losses += 0.005*torch.mean(D_left[l_flag] - 0.0)
            losses += 0.005*self.LossL1(left_hand_vertices_opt, left_hand_vertices)
        return losses, rotmat2d6(sbj_opt['right_hand_pose']), rotmat2d6(sbj_opt['left_hand_pose'])

    def load_data(self,args, ds_name):
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
        arms_rot_fix =  fullpose_with_fingers[:, :, arm_joints, :].to(self.device).float().reshape(batch_size, self.time, -1)
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
            # 'obj_v_template': batch['obj_v_template'],
            'upperbody_rot': upperbody_rot,
            'right_arm_rot': right_arm_rot,
            'rest_body_rot': rest_body_rot,
            'left_arm_rot': left_arm_rot,
            'arms_rot': arms_rot,
            'arms_rot_fix': arms_rot_fix,
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
            # 'lhand2rhand': lhand2rhand,
            'obj_transl': obj_transl,
            'obj_global_orient': obj_global_orient,
            # 'obj_bps': obj_bps,
            'right_hand_verts': right_hand_vert,
            'left_hand_verts': left_hand_vert,
            'verts': verts,
            'verts_obj': verts_obj,
            # 'contact': body_contact,
            'obj_vec': batch['obj_vec'].to(self.device),
            'sbj_id': batch['sbj_id'],
            'intent_vec': torch.randn((batch_size, 512)).to(self.device),
            'sentence_vec': batch['sentence_vec'].to(self.device),
            'sentence': batch['sentence'] ,
            'intent_embedding': batch['intent_embedding'].to(self.device),
            'obj_embedding': batch['obj_embedding'].to(self.device),
            'obj_name': batch['obj_name'][0],
            'intent': batch['intent'][0],
        }
        return data

    def test_all(self, ds_name='test'):
        self.arms_model.eval() 
        self.body_model.eval() 
        self.network.eval()
        test_loader = self.ds_test
        test_tqdm = tqdm(test_loader, desc= ds_name + ' {:.10f}'.format(0), leave=False, ncols=100)        
        for count, batch in enumerate(test_tqdm):           
            T = self.time 
            file_name =  batch['seq_name'][0].split('/')[-1][:-4] + '_' + str(count) + '.npz'
            data = self.load_batch(batch, batch_size=1)

            ARMS_out = [data['arms_rot'][:, 0, :].unsqueeze(1), data['arms_rot'][:, 1, :].unsqueeze(1), 
                            data['arms_rot'][:, 2, :].unsqueeze(1), data['arms_rot'][:, 3, :].unsqueeze(1)]    
            REST_BODY_out = [data['rest_body_rot'][:, 0, :].unsqueeze(1), data['rest_body_rot'][:, 1, :].unsqueeze(1), 
                            data['rest_body_rot'][:, 2, :].unsqueeze(1), data['rest_body_rot'][:, 3, :].unsqueeze(1)]
            OBJ_orient_out = [data['obj_global_orient'][:, 0, :].unsqueeze(1), data['obj_global_orient'][:, 1, :].unsqueeze(1), 
                            data['obj_global_orient'][:, 2, :].unsqueeze(1), data['obj_global_orient'][:, 3, :].unsqueeze(1)]    
            OBJ_trans_out = [data['obj_transl'][:, 0, :].unsqueeze(1), data['obj_transl'][:, 1, :].unsqueeze(1), 
                            data['obj_transl'][:, 2, :].unsqueeze(1), data['obj_transl'][:, 3, :].unsqueeze(1)]    
            FULL_POSE_out = [data['fullpose_with_fingers'][:, 0, :].unsqueeze(1), data['fullpose_with_fingers'][:, 1, :].unsqueeze(1), 
                            data['fullpose_with_fingers'][:, 2, :].unsqueeze(1), data['fullpose_with_fingers'][:, 3, :].unsqueeze(1)]    
            
            ARMS_fixjts = [data['arms_rot_fix'][:, 0, :].unsqueeze(1), data['arms_rot_fix'][:, 1, :].unsqueeze(1), 
                            data['arms_rot_fix'][:, 2, :].unsqueeze(1), data['arms_rot_fix'][:, 3, :].unsqueeze(1)]    
              
            intent_emb = data['intent_embedding']
            intent_vec = data['intent_vec']
            betas = data['betas'] 
            obj_vec = data['obj_vec']
            X0_arms = torch.cat([to_tensor(betas), to_tensor(intent_emb.squeeze(1))], dim=-1)
            X_arms = self.arms_model.enc_bn0(X0_arms)
            X_learned_arms = self.arms_model.enc_rb0(X_arms)
            X0_body = torch.cat([to_tensor(betas.squeeze(1)), to_tensor(obj_vec.squeeze(1)), to_tensor(intent_emb.squeeze(1))], dim=-1)
            X_body = self.body_model.enc_bn0(X0_body)
            X_learned_body = self.body_model.enc_rb0(X_body)
            '''Find out frame where object moves from rest'''
            for t_in in range(1, 4):
                obj_movement_threshold = torch.max(data['obj_transl'][:, 0, :] - data['obj_transl'][:, t_in, :])
                if obj_movement_threshold >= 0.001:
                    break # take t_in-1 as the frame with the correct grasp.
            obj_p = {
                'global_orient': d62aa(to_tensor(data['obj_global_orient'])[:, t_in-1: t_in]),
                'transl':  to_tensor(data['obj_transl'][:, t_in-1:t_in].squeeze(0)),
            }
            mesh_path = Mesh(filename=os.path.join(self.args1.data_path, object_mesh_path, data['obj_name'] +'.ply'))
            obj_mesh = Mesh(filename=mesh_path)
            np.random.seed(self.args1.seed)
            obj_v = np.array(obj_mesh.vertices)
            obj_p['v_template'] = to_tensor(obj_v)
            init_obj_vertex = self.object_model(**obj_p).vertices

            test_fullpose = data['fullpose_with_fingers'][:, t_in-1].clone()
            sbj_p = parms_6D2smplx_params(test_fullpose, data['transl'][:, t_in-1])   
            gender = data['gender']
            sbj_p['v_template'] = data['sbj_v_template'].to(self.device).float()
            if gender == 1:
                model_output = self.male_model(**sbj_p)
                verts_pred = model_output.vertices
                
            if gender == 2:
                model_output = self.female_model(**sbj_p)
                verts_pred = model_output.vertices
           
            right_hand_verts_pred = verts_pred[..., smplx_rhand_ver, :]
            left_hand_verts_pred = verts_pred[..., smplx_lhand_ver, :]
            
            if data['rhand_contact'].all():
                init_hand_vertex = right_hand_verts_pred
            elif data['rhand_contact'][:, 3]:
                init_hand_vertex = right_hand_verts_pred
            else:
                init_hand_vertex = left_hand_verts_pred
            init_obj_vertex = init_obj_vertex.squeeze(0)
            init_hand_vertex = init_hand_vertex.squeeze(0)
            init_obj_vertex1 = torch.repeat_interleave(init_obj_vertex, len(init_hand_vertex), dim=0)
            init_hand_vertex1 = init_hand_vertex.repeat(init_obj_vertex.shape[0], 1)
            diff = init_obj_vertex1 - init_hand_vertex1
            D = torch.norm(diff, dim=-1).float()
            x_opt = np.zeros((T, 6), dtype=np.float32)
            x_opt[0] = torch.cat((d62aa(data['obj_global_orient'][:, 0]), data['transl'][:, 0]), axis=-1).flatten().cpu().detach().numpy()
            x_opt[1] = torch.cat((d62aa(data['obj_global_orient'][:, 1]), data['transl'][:, 1]), axis=-1).flatten().cpu().detach().numpy()
            x_opt[2] = torch.cat((d62aa(data['obj_global_orient'][:, 2]), data['transl'][:, 2]), axis=-1).flatten().cpu().detach().numpy()
            x_opt[3] = torch.cat((d62aa(data['obj_global_orient'][:, 3]), data['transl'][:, 3]), axis=-1).flatten().cpu().detach().numpy()
            
            initial_params = {
            'right_hand_pose':  test_fullpose[:, right_hand_finger_joints],
            'left_hand_pose':  test_fullpose[:, left_hand_finger_joints],
            }
            
            for t in range(4, T):
                self.init_params(initial_params)
                obj_transl_past = torch.cat(OBJ_trans_out, dim=1)[:, t-4:t, :]
                obj_orient_past = torch.cat(OBJ_orient_out, dim=1)[:, t-4:t, :]
                arms_rot_prev = torch.cat(ARMS_out, dim=1)[:, t-4:t, :]
                arms_rot_out_ = self.arms_model.single_step_sample(args, arms_rot_prev, obj_transl_past, X_learned_arms)   
                ARMS_out.append(arms_rot_out_.unsqueeze(1))
                arms_rot_out = arms_rot_out_.reshape(1, -1, 6)
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
                
                arms_rot_body_in = arms_rot_body_in.reshape(1, -1)
                ARMS_fixjts.append(arms_rot_body_in.unsqueeze(1))
                arms_rot_prev_fixjts = torch.cat(ARMS_fixjts, dim=1)[:, t-4:t, :]
                
                obj_past = torch.cat((obj_transl_past, obj_orient_past), dim=-1) 
                prev_body_rot = torch.cat(REST_BODY_out, dim=1)[:, t-4:t, :]
                _, rest_body_out = self.body_model.grasp_pose(args, prev_body_rot, arms_rot_body_in, arms_rot_prev_fixjts, obj_past[:, -1, :], obj_past, X_learned_body)
                REST_BODY_out.append(rest_body_out.unsqueeze(1))
                test_fullpose = data['fullpose_with_fingers'][:, t_in-1].clone()
                
                #For initializing with GNet uncomment this block
                # gnet_batch = {
                #     'transl': batch['transl'][:, t],
                #     'transl_obj': batch['transl_obj'][:, t],
                #     'bps_obj_glob': batch['bps_obj_glob'][:, t],
                #     'betas': batch['betas'].squeeze(1)
                # }
                # rhand_finger_pose, lhand_finger_pose = self.gnet_infer(gnet_batch)          
                # rhand_finger_pose = rotmat2d6(rhand_finger_pose).reshape(1, -1, 6)
                # lhand_finger_pose = rotmat2d6(lhand_finger_pose).reshape(1, -1, 6)
                # test_fullpose[:, right_hand_finger_joints] = rhand_finger_pose
                # test_fullpose[:, left_hand_finger_joints] = lhand_finger_pose

                test_fullpose[:, arm_joints] = arms_rot_body_in.reshape(1, -1, 6)
                test_fullpose[:, rest_body_joints] = rest_body_out.reshape(1, -1, 6)
                sbj_p = parms_6D2smplx_params(test_fullpose, data['transl'][:, t])   
                gender = data['gender']
                sbj_p['v_template'] = data['sbj_v_template'].to(self.device).float()
                verts_pred = torch.zeros((1, 10475, 3)).to(self.device)
                if gender == 1:
                    smplx_model = self.male_model
                else:
                    smplx_model = self.female_model
                model_output = smplx_model(**sbj_p)
                verts_pred = model_output.vertices
                right_hand_verts_pred = verts_pred[0, smplx_rhand_ver, :]
                left_hand_verts_pred = verts_pred[0, smplx_lhand_ver, :]
                do_optimization = True
                rhand_flag = False
                if do_optimization:
                    if data['rhand_contact'][:, t]:
                        rhand_flag = True
                        res = least_squares(self.opt_model, x_opt[t-1], max_nfev=None, args=(obj_v, right_hand_verts_pred.cpu().detach().numpy(), D.cpu().detach().numpy()))          # try this with x_opt[ti-1] and x_opt[3] fixed
                        x_opt[t] = res.x
                    else:
                        res = least_squares(self.opt_model, x_opt[t-1], max_nfev=None, args=(obj_v, left_hand_verts_pred.cpu().detach().numpy(), D.cpu().detach().numpy()))
                        x_opt[t] = res.x

                    obj_opt = {
                        'global_orient': torch.from_numpy(x_opt[t, :3]).unsqueeze(0).to(device),
                        'transl': torch.from_numpy(x_opt[t, 3:6]).unsqueeze(0).to(device),
                        'v_template': to_tensor(obj_v),
                    }
                    obj_opt_vert = self.object_model(**obj_opt).vertices
                    for itr in range(self.opt_num_iters):
                        self.optimizer.zero_grad()
                        curr_lr = self.lr * pow(0.99999, itr)
                        for param_group in self.optimizer.param_groups:                
                            param_group['lr'] = curr_lr
                        losses, r_fingers, l_fingers = self.calc_loss_fingers(smplx_model, sbj_p, obj_opt_vert, rhand_flag)
                        losses.backward(retain_graph=True)
                        self.optimizer.step()
                    
                    test_fullpose[0, left_hand_finger_joints] = l_fingers
                    test_fullpose[0, right_hand_finger_joints] = r_fingers

                    OBJ_trans_out.append(to_tensor(x_opt[t, 3:]).float().reshape(1, -1, 3))
                    OBJ_orient_out.append(aa2d6(to_tensor(x_opt[t, :3].reshape(-1, 3))).unsqueeze(0))

                else:
                    OBJ_trans_out.append(data['obj_transl'][:, t, :].unsqueeze(1))
                    OBJ_orient_out.append(data['obj_global_orient'][:, t, :].unsqueeze(1))
                FULL_POSE_out.append(test_fullpose.unsqueeze(1)) 

            output_fullpose = torch.cat(FULL_POSE_out, dim=1)
            sbj_p_output = batch_parms_6D2smplx_params(output_fullpose, data['transl'][:, :t+1])
            output_obj_trans = torch.cat(OBJ_trans_out, dim=1).squeeze(0)
            output_obj_orient = torch.cat(OBJ_orient_out, dim=1).squeeze(0)
            obj_p_output = {
                'transl': output_obj_trans,
                'global_orient': d62aa(output_obj_orient)
            }
            output = {
                'obj_p': obj_p_output,
                'sbj_p': sbj_p_output,
                'sequence_name':batch['seq_name'][0],
                'sbj_vtemp':data['sbj_v_template'],
                'gender': gender, 
                'obj_name': batch['obj_name'][0]
            }
            outfname = makepath(os.path.join(self.args2.save_dir, self.book2.name.name , ds_name, file_name), isfile=True)
            torch.save(output, outfname)


if __name__ == '__main__':
    args = argparseNloop()
    ds_name = 'test'
    model_tester = Tester(args=args, ds_name=ds_name)
    print(" **Inititalization done**")
    model_tester.test_all(ds_name)
