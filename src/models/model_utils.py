# -*- coding: utf-8 -*-
#
# Copyright (C) 2022 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#

import torch
import math
import torch.nn as nn
import torch.nn.init as nninit
from src.tools.transformations import aa2d6, d62aa, d62rotmat


class PositionalEncoder(nn.Module):
    """
    Positional encoder: Encodes the joint index prior to processing all joints with an attention module
    """

    def __init__(self, d_model, seq_len):
        super().__init__()
        self.d_model = d_model

        pe = torch.zeros(seq_len, d_model)
        for pos in range(seq_len):
            for i in range(0, d_model-1, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        x = x + 1e-3 * self.pe[:, :seq_len]

        return x

smplx_parents =[-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9,  9, 12, 13, 14,
                16, 17, 18, 19, 15, 15, 15, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34,
                35, 20, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50, 21, 52,
                53]
def smplx_loc2glob(local_pose):

    bs = local_pose.shape[0]
    local_pose = local_pose.view(bs, -1, 3, 3)
    global_pose = local_pose.clone()

    for i in range(1,len(smplx_parents)):
        global_pose[:,i] = torch.matmul(global_pose[:, smplx_parents[i]], global_pose[:, i].clone())

    return global_pose.reshape(bs,-1,3,3)


@torch.no_grad()
def fullpose2smplx_params_aa(pose,trans):
    global_orient = pose[..., :3]
    body_pose = pose[..., 3:66]
    jaw_pose  = pose[..., 66:69]
    leye_pose = pose[..., 69:72]
    reye_pose = pose[..., 72:75]
    left_hand_pose = pose[..., 75:120]
    right_hand_pose = pose[..., 120:]

    body_parms = {'global_orient': global_orient, 'body_pose': body_pose,
                  'jaw_pose': jaw_pose, 'leye_pose': leye_pose, 'reye_pose': reye_pose,
                  'left_hand_pose': left_hand_pose, 'right_hand_pose': right_hand_pose,
                  'transl': trans}
    return body_parms

@torch.no_grad()
def fullpose2smplx_params_rotmat(pose,trans):
    
    global_orient = pose[:, 0:1]
    body_pose = pose[:, 1:22]
    jaw_pose  = pose[:, 22:23]
    leye_pose = pose[:, 23:24]
    reye_pose = pose[:, 24:25]
    left_hand_pose = pose[:, 25:40]
    right_hand_pose = pose[:, 40:]

    body_parms = {'global_orient': global_orient, 'body_pose': body_pose,
                  'jaw_pose': jaw_pose, 'leye_pose': leye_pose, 'reye_pose': reye_pose,
                  'left_hand_pose': left_hand_pose, 'right_hand_pose': right_hand_pose,
                  'transl': trans}
    return body_parms

def parms_6D2smplx_params(pose,trans, d62rot=True):
    b = pose.shape[0]
    trans = trans.reshape(b, -1)
    if d62rot:
        pose = d62rotmat(pose)
    pose = pose.reshape([b, -1, 3, 3])
    body_parms = fullpose2smplx_params_rotmat(pose, trans)
    return body_parms

def obj_6D2obj_model_params(obj_pose, obj_trans, is_d62aa=True):
    b = obj_pose.shape[0] 
    obj_trans = obj_trans.reshape(b, -1)
    if is_d62aa:
        obj_pose = d62aa(obj_pose).reshape(b, -1)
    obj_parms = {
        'global_orient': obj_pose,
        'transl': obj_trans
    }
    return obj_parms

def batch_parms_6D2smplx_params(pose, trans, d62rot=True):
    b = pose.shape[0]
    t = pose.shape[1]
    trans = trans.reshape(b*t, -1)
    if d62rot:
        pose = d62rotmat(pose)
    pose = pose.reshape([b*t, -1, 3, 3])
    body_parms = fullpose2smplx_params_rotmat(pose,trans)
    return body_parms

def batch_obj_6D2obj_model_params(obj_pose, obj_trans, is_d62aa=True):
    b = obj_pose.shape[0]
    t = obj_pose.shape[1]
    obj_trans = obj_trans.reshape(b*t, -1)
    if is_d62aa:
        obj_pose = d62aa(obj_pose).reshape(b*t, -1)
    obj_pose = obj_pose.reshape(b*t, -1)
    obj_parms = {
        'global_orient': obj_pose,
        'transl': obj_trans
    }
    
    return obj_parms



def parms_6D2full(pose,trans, d62rot=True, dtype = torch.float32):

    bs = trans.shape[0]

    if d62rot:
        pose = d62rotmat(pose)
    pose = pose.reshape([bs, -1, 3, 3]).to(dtype)

    body_parms = fullpose2smplx_params_rotmat(pose,trans)
    body_parms['fullpose_rotmat'] = pose

    return body_parms

