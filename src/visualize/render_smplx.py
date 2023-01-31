import sys
sys.path.append('.')
sys.path.append('..')
import smplx
import torch
import trimesh
import numpy as np
import os, glob
from smplx import SMPLXLayer
from src.tools.argUtils import argparseNloop
from src.tools.objectmodel import ObjectModel
from src.tools.meshviewer import Mesh, MeshViewer, colors
from src.tools.utils import makepath, to_cpu
from src.tools.transformations import euler, to_tensor, rotmat2aa
from src.data_loader.dataloader import *
from scipy import interpolate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
smplx_rhand_ver = np.load(os.path.join('..', 'DATASETS', 'GRAB', 'tools','smplx_correspondence', 'rhand_smplx_ids.npy'))
object_mesh_path = os.path.join('..', 'DATASETS', 'GRAB', 'tools','object_meshes', 'new_contact_meshes')
subject_mesh_path = os.path.join('..', 'DATASETS', 'GRAB', 'tools','subject_meshes')

def visualize_sequences(args, dir_path):
    ds_name = dir_path.split('\\')[-1]
    wish = True
    mv = MeshViewer(width=800, height=800, bg_color = [1.0, 1.0, 1.0, 0.6], offscreen=wish)
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = euler([80, -15, 10], 'xzx')
    camera_pose[:3, 3] = np.array([-0.5, -2.5, 0])
    mv.update_camera_pose(camera_pose)
    print('Start loading data.')
    all_seqs = glob.glob(dir_path + '\\*.npz')  
    print('Completed loading data.')
    for i in range(0, len(all_seqs)):
        vis_sequence(args, all_seqs[i], i, dir_path, mv, ds_name, wish)
    if not wish:
        mv.close_viewer()

def vis_sequence(args, seq_data, count, dir_path, mv, ds_name, save_png):
    sequence = torch.load(seq_data)
    sequence_name = seq_data.split('\\')[-1][:-4]
    sbj_id = sequence_name.split('_')[0]
    obj_name = sequence_name.split('_')[1]
    intent_name = sequence_name.split('_')[2]
    if sbj_id =='s1' or sbj_id =='s2' or sbj_id =='s8' or sbj_id =='s9' or sbj_id =='s10':
        gender = 'male'
    else:
        gender = 'female'
    sbj_p = sequence['sbj_p']
    T = sbj_p['transl'].shape[0]
    sbj_vtemp = to_tensor(np.array(Mesh(filename=os.path.join(subject_mesh_path, gender, sbj_id + '.ply')).vertices))
    np.random.seed(args.seed)
    verts_sample_id = np.random.choice(sbj_vtemp.shape[0], 900, replace=False)
    sbj_p['v_template'] = sbj_vtemp
    if sbj_p['global_orient'].shape == (T, 3, 3) or sbj_p['global_orient'].shape == (T, 1, 3, 3):
        sbj_m = SMPLXLayer(
                model_path=os.path.join(args.model_path, 'smplx'),
                gender=gender,
                num_pca_comps=45,
                flat_hand_mean=True).to(device)
    elif sbj_p['global_orient'].shape == (T, 3):
        sbj_m = smplx.create(model_path=args.model_path,
                             model_type='smplx',
                             gender=gender,
                             num_pca_comps=45,
                             v_template=sbj_vtemp,
                             batch_size=T).to(device)
    verts_sbj = to_cpu(sbj_m(**sbj_p).vertices)
    obj_p = sequence['obj_p']
    if obj_p['global_orient'].shape == (T, 3, 3):
        obj_p['global_orient'] = rotmat2aa(obj_p['global_orient'])
    elif obj_p['global_orient'].shape == (T, 3):
        obj_p['global_orient'] = obj_p['global_orient'].squeeze()
    obj_p['transl'] = obj_p['transl'].squeeze()
    obj_mesh = Mesh(filename=os.path.join(object_mesh_path, obj_name+'.ply'))
    obj_p['v_template'] = torch.from_numpy(np.array(obj_mesh.vertices)).to(device).float()
    obj_m = ObjectModel().to(device)
    verts_obj = to_cpu(obj_m(**obj_p).vertices)
    faces_obj = obj_mesh.faces
    verts_all = np.concatenate((verts_obj, verts_sbj), axis=1)
    new_faces_sbj = sbj_m.faces + verts_obj.shape[1]
    faces_all = np.concatenate((faces_obj, new_faces_sbj), axis =0)
    # linear interpolation
    T1 = 2*T
    verts_x_interp =np.zeros((T1, verts_all.shape[1]))
    verts_y_interp =np.zeros((T1, verts_all.shape[1]))
    verts_z_interp =np.zeros((T1, verts_all.shape[1]))
    x = np.linspace(0, T-1 ,T)
    x_new = np.linspace(0, T-1 ,T1)
    for v1 in range(0, verts_all.shape[1]):
        verts_sbj_x = verts_all[:, v1, 0]
        verts_sbj_y = verts_all[:, v1, 1]
        verts_sbj_z = verts_all[:, v1, 2]
        f_x = interpolate.interp1d(x, verts_sbj_x, kind = 'linear')
        f_y = interpolate.interp1d(x, verts_sbj_y, kind = 'linear')
        f_z = interpolate.interp1d(x, verts_sbj_z, kind = 'linear')
        verts_x_interp[:, v1] = f_x(x_new)
        verts_y_interp[:, v1] = f_y(x_new)
        verts_z_interp[:, v1] = f_z(x_new)
    verts_x_interp = torch.from_numpy(verts_x_interp).unsqueeze(2)
    verts_y_interp = torch.from_numpy(verts_y_interp).unsqueeze(2)
    verts_z_interp = torch.from_numpy(verts_z_interp).unsqueeze(2)
    verts_interp = torch.cat((verts_x_interp, verts_y_interp, verts_z_interp), dim=-1)
    verts_all = torch.cat((torch.from_numpy(verts_all[0]).unsqueeze(0), verts_interp), dim=0)
    T = T1
    color = np.ones([verts_all.shape[0], verts_all.shape[1], 4]) * [1.0, 0.0, 0.0, 1]
    color[:, -10475:] = (1.0, 0.75, 0.8, 1)
    skip_frame = 1
    for frame in range(0,T, skip_frame):
        all_mesh = Mesh(vertices=verts_all[frame], faces=faces_all, vc=color[frame], smooth=True)
        mv.set_static_meshes([all_mesh])
        if save_png:
            seq_render_path = makepath(os.path.join(dir_path, seq_data.split('\\')[-1][:-4] + 'in', str(frame) + '.png'),  isfile=True)
            mv.save_snapshot(seq_render_path)
    print(intent_name)

if __name__ == '__main__':
    args = argparseNloop()
    visualize_sequences(args, dir_path='visualize_samples') #change the dir_path to where the generated '.npy' files are located.