import numpy as np
import os
import pickle as pkl
import random
import sys
sys.path.append('.')
sys.path.append('..')
import torch
import torch.nn as nn
import torch.nn.functional as F
from cmath import nan
from bps_torch.bps import bps_torch
from smplx import SMPLXLayer
from src.tools.objectmodel import ObjectModel
from src.tools.utils import to_tensor
from src.models.model_utils import *
from src.models.attention import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_path = ''
smplx_rhand_ver = np.load(os.path.join(base_path, '..', 'DATASETS', 'GRAB', 'tools', 'smplx_correspondence', 'rhand_smplx_ids.npy'))
smplx_lhand_ver = np.load(os.path.join(base_path, '..', 'DATASETS', 'GRAB', 'tools', 'smplx_correspondence', 'lhand_smplx_ids.npy'))
object_mesh_path = os.path.join('tools','object_meshes', 'contact_meshes')
subject_mesh_path = os.path.join('tools','subject_meshes')

used_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
right_hand_finger_joints = [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]
left_hand_finger_joints = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
reduce_right_hand_finger_joints = [40, 43, 46, 49, 52]
reduce_left_hand_finger_joints = [25, 28, 31, 34, 37]
upperbody_joints = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
right_arm_joints = [17, 19, 21]
arm_joints = [16, 17, 18, 19, 20, 21]
parents_idx_arms_joints = [0, 1, 0, 1, 2, 3]
left_arm_joints = [16, 18, 20]
lowerbody_joints = [0, 1, 2, 4, 5, 7, 8, 10, 11]
rest_body_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
parents_idx_bdy_joints = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 12, 12]

def initialize_weights(m):
    std_dev = 0.02
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=std_dev)
        if m.bias is not None:
            nn.init.normal_(m.bias, std=std_dev)
    elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, std=std_dev)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias, std=std_dev)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, std=std_dev)  
        if m.bias is not None:
            nn.init.normal_(m.bias, std=std_dev)  
       
def masking(frame_len, N):
    p = list(np.ones(frame_len))
    if N < frame_len:
        for i in range(N-1, frame_len, (N)):
            (p)[i] = (p)[i] * (0)
    return p

class BatchFlatten(nn.Module):
    def __init__(self):
        super(BatchFlatten, self).__init__()
        self._name = 'batch_flatten'

    def forward(self, x):
        return x.view(-1, x.shape[-1])


class ResBlock(nn.Module):
    def __init__(self,
                 Fin,
                 Fout,
                 n_neurons=256):

        super(ResBlock, self).__init__()
        self.Fin = Fin
        self.Fout = Fout
        self.fc1 = nn.Linear(Fin, n_neurons)
        self.bn1 = nn.BatchNorm1d(n_neurons)
        self.do = nn.Dropout(p=.1, inplace=False)
        self.fc2 = nn.Linear(n_neurons, Fout)
        self.bn2 = nn.BatchNorm1d(Fout)

        if Fin != Fout:
            self.fc3 = nn.Linear(Fin, Fout)

        self.ll = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, final_nl=True):
        Xin = x if self.Fin == self.Fout else self.ll(self.fc3(x))

        Xout = self.fc1(x)
        Xout = self.bn1(Xout)
        Xout = self.ll(Xout)

        Xout = self.fc2(self.do(Xout))
        Xout = self.bn2(Xout)
        Xout = Xin + Xout

        if final_nl:
            return self.ll(Xout)
        return Xout

class Interaction_Prior_CVAE(nn.Module): 
    def __init__(self, args, device, time):
        super(Interaction_Prior_CVAE, self).__init__()
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        self.latentD = args.latentD
        self.time = time
        self.device = device
        self.weight_manifoldloss = args.loss_manifold
        self.weight_kldiv = args.loss_kldiv
        self.weight_contact = args.loss_contact
        self.weight_verts = args.loss_verts
        self.weight_dist = args.loss_dist
        self.weight_reconstruction = args.loss_reconstruction
        self.weight_angleprior = args.loss_angle_prior
        self.hidden = 1024
        self.learned_features = 400
        self.hand_rot = 8*6
        self.body_rot = 6*6
        self.obj_bps = 1024
        self.betas = 10
        self.verts = 1000
        self.obj_trans = 3
        self.intent_vectors = 512
        self.vposer_dim = 0
        self.bps_torch = bps_torch(n_bps_points=512, random_seed=args.seed)
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
        self.enc_bn0 = nn.BatchNorm1d(self.betas + self.intent_vectors )
        self.enc_rb0 = ResBlock(self.betas + self.intent_vectors , self.learned_features)
        self.enc_bn1 = nn.BatchNorm1d( self.body_rot + self.obj_trans + self.learned_features)
        self.enc_rb1 = ResBlock( self.body_rot + self.obj_trans + self.learned_features, 4 * self.hidden)
        self.enc_bn1_past = nn.BatchNorm1d(4 * (self.body_rot + self.obj_trans) + self.learned_features)
        self.enc_rb1_past = ResBlock(4 * (self.body_rot + self.obj_trans) + self.learned_features, 4 * self.hidden)
        self.enc_rb2 = ResBlock(4 * self.hidden + (self.body_rot + self.obj_trans ) + self.learned_features, self.hidden) 
        self.enc_rb2_past = ResBlock(4 * self.hidden + 4 * (self.body_rot + self.obj_trans) + self.learned_features, self.hidden)
        self.mu_present = nn.Linear(self.hidden, self.latentD)
        self.sigma_present = nn.Linear(self.hidden, self.latentD)
        self.mu_past = nn.Linear(self.hidden, self.latentD)
        self.sigma_past = nn.Linear(self.hidden, self.latentD)
        self.dec_bn1 = nn.BatchNorm1d(self.learned_features + self.latentD + 4 * (self.body_rot + self.obj_trans) ) 
        self.dec_rb1 = ResBlock(self.learned_features + self.latentD + 4 * (self.body_rot + self.obj_trans), 2 * self.hidden)
        self.dec_rb2 = ResBlock(2 * self.hidden + self.latentD + 4 * (self.body_rot + self.obj_trans) + self.learned_features , self.body_rot)
        
    
    def grasp_pose(self, args, body_rot, body_rot_past, obj_transl, obj_transl_past, X_learned, is_train):
        bs =X_learned.shape[0]
        '''Encode Present state'''
        X_ = torch.cat([body_rot.reshape(bs, -1), obj_transl.reshape(bs, -1), X_learned], dim=-1)
        X0 = self.enc_bn1(X_)
        X1  = self.enc_rb1(X0, True)
        X  = self.enc_rb2(torch.cat([X0, X1], dim=-1), True)
        mu_present = self.mu_present(X)
        sig_present = self.sigma_present(X)
        e = torch.randn(mu_present.shape).to(self.device)
        kldiv_loss_present = self.loss_kldiv(mu_present, sig_present, self.weight_kldiv)
        z_present = e*sig_present + mu_present
        '''Encode past state'''
        Xp = torch.cat([body_rot_past.reshape(bs, -1), obj_transl_past.reshape(bs, -1), X_learned], dim=-1)
        X0p = self.enc_bn1_past(Xp)
        Xp = self.enc_rb1_past(X0p, True)
        Xp = self.enc_rb2_past(torch.cat([X0p, Xp], dim=-1), True)
        mu_past = self.mu_past(Xp)
        sig_past = self.sigma_past(Xp)
        kldiv_loss_past = self.loss_kldiv(mu_past, sig_past, self.weight_kldiv)
        e = torch.randn(mu_past.shape).to(mu_past.device)
        z_past = e*sig_past + mu_past
        if is_train:
            z = z_present if random.random() > 0.5 else z_past      # Can do a random pick     
        else:
            z = z_past
        '''Decode using z and previous states'''
        Y0 = torch.cat([X_learned, z, body_rot_past.reshape(bs, -1), obj_transl_past.reshape(bs, -1)], dim=-1)
        Y1 = self.dec_bn1(Y0)
        Y2 = self.dec_rb1(Y1)
        Y3 = self.dec_rb2(torch.cat((Y0, Y2), dim=-1))
        body_pose_out = Y3[:, -self.body_rot:] + body_rot_past[:,-1,:].reshape(bs, -1)
        internal_loss = [kldiv_loss_present, kldiv_loss_past]
        return internal_loss, body_pose_out
   
    def forward(self, args, t, mask, data, up_body_prev, is_train=True):
        bs = up_body_prev.shape[0]
        up_body_pose = data['arms_rot'][:, :(t+1), :]
        obj_transl = data['obj_transl'][:, :(t+1), :]
        intent_vec = data['intent_embedding']
        betas = data['betas']
        body_pose_past = (1-mask[t]) * up_body_pose[:, t-4:t, :] + (mask[t]) * up_body_prev
        obj_transl_past = obj_transl[:, t-4:t, :] 
        X0 = torch.cat([to_tensor(betas.squeeze(1)), to_tensor(intent_vec.squeeze(1))], dim=-1)
        X = self.enc_bn0(X0)
        X_learned = self.enc_rb0(X)
        internal_loss, up_body_out = self.grasp_pose(args, up_body_pose[:, t, :], body_pose_past, obj_transl[:, t, :], obj_transl_past, X_learned, is_train)
        rec_loss = self.weight_reconstruction * self.loss_reconstruction(body_pose_past[:, -1, :],
                                                    up_body_pose[:, t-1, :], up_body_out, up_body_pose[:, t, :], bs) 
        internal_loss.append(rec_loss)
        return internal_loss, up_body_out.reshape(bs, 1, -1)
    
    def loss_reconstruction(self, data_past, data_past_gt, data_present, data_present_gt, b):
        return F.l1_loss(data_present, data_present_gt) + F.l1_loss(data_present - data_past, data_present_gt - data_past_gt) 
    
    def loss_kldiv(self, mu, sig, wkl):
        L_KL = -0.5 * torch.mean(1 + torch.log(sig*sig) - mu.pow(2) - sig.pow(2))
        return wkl * L_KL
    
    def single_step_sample(self, args, body_rot_past, obj_transl_past, X_learned):
        bs = body_rot_past.shape[0]
        Xp = torch.cat([body_rot_past.reshape(bs, -1), obj_transl_past.reshape(bs, -1), X_learned], dim=-1)
        X0p = self.enc_bn1_past(Xp)
        Xp = self.enc_rb1_past(X0p, True)
        Xp = self.enc_rb2_past(torch.cat([X0p, Xp], dim=-1), True)
        mu_past = self.mu_past(Xp)
        sig_past = self.sigma_past(Xp)
        e = torch.randn(mu_past.shape).to(mu_past.device)
        z = e*sig_past + mu_past 
        '''Decode using z and previous states'''
        Y0 = torch.cat([X_learned, z, body_rot_past.reshape(bs, -1), obj_transl_past.reshape(bs, -1)], dim=-1)
        Y1 = self.dec_bn1(Y0)
        Y2 = self.dec_rb1(Y1)
        Y3 = self.dec_rb2(torch.cat((Y0, Y2), dim=-1))
        body_pose_out = Y3[:, -self.body_rot:] + body_rot_past[:,-1,:].reshape(bs, -1)
        return body_pose_out
    
    def sample(self, args, T, data):
        ARMS_out = [data['arms_rot'][:, 0, :].unsqueeze(1), data['arms_rot'][:, 1, :].unsqueeze(1), 
                            data['arms_rot'][:, 2, :].unsqueeze(1), data['arms_rot'][:, 3, :].unsqueeze(1)]    
        intent_vec = data['intent_embedding']
        betas = data['betas'] 
        X0 = torch.cat([to_tensor(betas), to_tensor(intent_vec.squeeze(1))], dim=-1)
        X = self.enc_bn0(X0)
        X_learned = self.enc_rb0(X)
        for t in range(4, T):
            obj_transl_past = data['obj_transl'][:, t-4:t, :]
            arms_rot_prev = torch.cat(ARMS_out, dim=1)[:, t-4:t, :]
            arms_rot_out = self.single_step_sample(args, arms_rot_prev, obj_transl_past, X_learned)   
            ARMS_out.append(arms_rot_out.unsqueeze(1))
        return torch.cat(ARMS_out, dim=1).squeeze(0)

class Interaction_Prior_Posterior_CVAE_clip(nn.Module): 
    def __init__(self, args, device, time):
        super(Interaction_Prior_Posterior_CVAE_clip, self).__init__()
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        self.latentD = args.latentD
        self.time = time
        self.device = device
        self.weight_manifoldloss = args.loss_manifold
        self.weight_kldiv = args.loss_kldiv
        self.weight_contact = args.loss_contact
        self.weight_verts = args.loss_verts
        self.weight_dist = args.loss_dist
        self.weight_reconstruction = args.loss_reconstruction
        self.weight_angleprior = args.loss_angle_prior
        self.hidden = 250
        self.learned_features = 70
        self.body_rot = 16*6
        self.betas = 10
        self.verts = 1000
        self.arms_rot = 6*6
        self.obj_dim = 9
        self.intent_vectors = 512
        self.obj_vec = 54
        self.vposer_dim = 0
        self.bps_torch = bps_torch(n_bps_points=512, random_seed=args.seed)
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
        self.enc_bn0 = nn.BatchNorm1d(self.betas + self.intent_vectors + self.obj_vec)
        self.enc_rb0 = ResBlock(self.betas + self.intent_vectors + self.obj_vec, self.learned_features)
        self.enc_bn1_past = nn.BatchNorm1d(4 * (self.body_rot + self.arms_rot + self.obj_dim) + self.learned_features + self.arms_rot + self.obj_dim)
        self.enc_rb1_past = ResBlock(4 * (self.body_rot + self.arms_rot + self.obj_dim ) + self.learned_features + self.arms_rot + self.obj_dim, 2 * self.hidden)
        self.enc_rb2_past = ResBlock(4 * (self.body_rot + self.arms_rot + self.obj_dim ) + self.learned_features + self.arms_rot + self.obj_dim + 2 * self.hidden, self.hidden)
        self.mu_past = nn.Linear(self.hidden, self.latentD)
        self.sigma_past = nn.Linear(self.hidden, self.latentD)
        self.dec_bn1 = nn.BatchNorm1d(self.learned_features + self.latentD + 4 * (self.body_rot + self.arms_rot + self.obj_dim) + self.arms_rot + self.obj_dim) 
        self.dec_rb1 = ResBlock(self.learned_features + self.latentD + 4 * (self.body_rot + self.arms_rot + self.obj_dim) + self.arms_rot + self.obj_dim, 2 * self.hidden)
        self.dec_rb2 = ResBlock(2 * self.hidden + self.latentD + 4 * (self.body_rot + self.arms_rot + self.obj_dim) + self.arms_rot + self.obj_dim + self.learned_features , self.body_rot)
        self.spatial_positional_encoder_body_arm_rot = PositionalEncoder(d_model=6, seq_len=16)
        model_attention = {'dim_k': 16, 'dim_v': 16, 'num_layers': 4, 'num_heads': 4}
        self.attention_model = AttentionModel(6, 0.1, model_attention, num_out_joints=16)
        self.activation = nn. LeakyReLU()

    def grasp_pose(self, args, body_rot_past, arms_rot, arms_rot_past, obj_, obj_past, X_learned):
        bs =X_learned.shape[0]
        '''Encode Attention module state'''
        attention_feature = to_tensor(torch.zeros(bs, 4, 16, 6))
        input_joint_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        for ti in range(4):
            body_rot_pe = self.spatial_positional_encoder_body_arm_rot(body_rot_past[:, ti].reshape(bs, -1, 6))
            attention_feature[:, ti] = self.attention_model(body_rot_pe[:, tuple(input_joint_indices), :], body_rot_pe)
        '''Encode past state'''
        Xp = torch.cat([attention_feature.reshape(bs, -1), arms_rot_past.reshape(bs, -1), arms_rot.reshape(bs, -1), obj_.reshape(bs, -1), obj_past.reshape(bs, -1), X_learned], dim=-1)
        X0p = self.enc_bn1_past(Xp)
        X1p = self.enc_rb1_past(X0p, True)
        X2p = self.enc_rb2_past(torch.cat([X0p, X1p], dim=-1), True)
        mu_past = self.mu_past(X2p)
        sig_past = self.sigma_past(X2p)
        kldiv_loss_past = self.loss_kldiv(mu_past, sig_past, self.weight_kldiv)
        e = torch.randn(mu_past.shape).to(mu_past.device)
        z_past = e*sig_past + mu_past
        '''Decode using z and previous states'''
        Y0 = torch.cat((z_past, Xp), dim=-1)
        Y1 = self.dec_bn1(Y0)
        Y2 = self.dec_rb1(Y1)
        Y3 = self.dec_rb2(torch.cat((Y0, Y2), dim=-1))
        body_pose_out = Y3[:, -self.body_rot:] 
        internal_loss = [kldiv_loss_past]
        return internal_loss, body_pose_out
   
    
    def forward(self, args, t, mask, data, up_body_prev, is_train=True):
        bs = up_body_prev.shape[0]
        rest_body_pose = data['rest_body_rot']
        obj_ = torch.cat((data['obj_transl'], data['obj_global_orient']), dim=-1) 
        intent_vec = data['intent_embedding']
        obj_vec = data['obj_vec']
        betas = data['betas']
        body_pose_past = (1-mask[t]) * rest_body_pose[:, t-4:t, :] + (mask[t]) * up_body_prev
        arms_rot_past = data['arms_rot'][:, t-4:t, :] 
        obj_past = obj_[:, t-4:t, :] 
        X0 = torch.cat([to_tensor(betas.squeeze(1)), to_tensor(obj_vec.squeeze(1)), to_tensor(intent_vec.squeeze(1))], dim=-1)
        X = self.enc_bn0(X0)
        X_learned = self.enc_rb0(X)
        internal_loss, body_out = self.grasp_pose(args, body_pose_past, data['arms_rot'][:, t, :], arms_rot_past, obj_[:, t, :], obj_past, X_learned)
        rec_loss = self.weight_reconstruction * self.loss_reconstruction(body_pose_past[:, -1, :],
                                                    rest_body_pose[:, t-1, :], body_out, rest_body_pose[:, t, :], bs) 
        internal_loss.append(rec_loss)
        fullbody_with_fingers = data['fullpose_with_fingers'][:, t].clone()
        fullbody_with_fingers[:, rest_body_joints, :] = body_out.reshape(bs, -1, 6)
        sbj_p = parms_6D2smplx_params(fullbody_with_fingers, data['transl'][:, t, :])
        gender = data['gender']
        sbj_v_template = data['sbj_v_template'].to(self.device).float()
        males = gender == 1
        females = ~males
        joints_out = torch.zeros((bs, 127, 3)).to(self.device)
        if sum(females) > 0:
            f_params = {k: v[females] for k, v in sbj_p.items()}
            f_params['v_template'] = sbj_v_template[females]
            f_output = self.female_model(**f_params)
            joints_out[females] = f_output.joints
        if sum(males) > 0:
            m_params = {k: v[males] for k, v in sbj_p.items()}
            m_params['v_template'] = sbj_v_template[males]
            m_output = self.male_model(**m_params)
            joints_out[males] = m_output.joints
        joints_loss =  self.weight_dist * F.l1_loss(data['joints'][:, t, :16], joints_out[:, :16])
        internal_loss.append(joints_loss)
        return internal_loss, body_out.reshape(bs, 1, -1)
    
    def loss_reconstruction(self, data_past, data_past_gt, data_present, data_present_gt, b):
        return F.l1_loss(data_present, data_present_gt) + F.l1_loss(data_present - data_past, data_present_gt - data_past_gt) 
    
    def loss_kldiv(self, mu, sig, wkl):
        L_KL = -0.5 * torch.mean(1 + torch.log(sig*sig) - mu.pow(2) - sig.pow(2))
        return wkl * L_KL
    
    def sample(self, args, T, data, arms_rot):
        REST_BODY_out = [data['rest_body_rot'][:, 0, :].unsqueeze(1), data['rest_body_rot'][:, 1, :].unsqueeze(1), 
                            data['rest_body_rot'][:, 2, :].unsqueeze(1), data['rest_body_rot'][:, 3, :].unsqueeze(1)]
        obj_ = torch.cat((data['obj_transl'], data['obj_global_orient']), dim=-1) 
        intent_vec = data['intent_embedding']
        obj_vec = data['obj_vec']
        betas = data['betas']
        
        X0 = torch.cat([to_tensor(betas.squeeze(1)), to_tensor(obj_vec.squeeze(1)), to_tensor(intent_vec.squeeze(1))], dim=-1)
        X = self.enc_bn0(X0)
        X_learned = self.enc_rb0(X)
        for t in range(4, T):
            prev_body_rot = torch.cat(REST_BODY_out, dim=1)[:, t-4:t, :]
            arms_rot_past = arms_rot[:, t-4:t, :] 
            obj_past = obj_[:, t-4:t, :] 
            _, rest_body_out = self.grasp_pose(args, prev_body_rot, arms_rot[:, t, :], arms_rot_past, obj_[:, t-1, :], obj_past, X_learned)
            REST_BODY_out.append(rest_body_out.unsqueeze(1))
        return torch.cat(REST_BODY_out, dim=1).squeeze(0)