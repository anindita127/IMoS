import glob
import os
import shutil
import sys
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
from cmath import nan
from torch import optim
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
from src.data_loader.dataloader import *
from src.tools.argUtils import argparseNloop
from src.tools.BookKeeper import *
from src.tools.utils import makepath
from src.models.action_rec import LSTM_Action_Classifier
from src.models.model import *
from tensorboardX import SummaryWriter

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

action_class = ['offhand','pass', 'lift', 'drink', 'brush', 'eat', 'peel', 'takepicture', 'see', 'wear', 'play', 'clean', 'browse', 'inspect', 'stamp',
                'pour', 'use', 'switchON', 'cook', 'toast', 'staple', 'squeeze', 'set', 'open', 'chop', 'screw', 'call', 'shake', 'fly']
action_classes = {}
for x in range(len(action_class)):
    action_classes[action_class[x]] = x   

def one_hot_vectors(action_list):
    vec = []
    for action in action_list:
        arr = list(np.zeros(len(action_class), dtype = int))
        arr[action_classes[action]] = 1
        vec.append(arr)
    return np.array(vec)

class Trainer:
    def __init__(self,args):
        torch.manual_seed(args.seed)
        args.batch_size = 1
        
        self.model_path = args.model_path
        makepath(args.work_dir, isfile=False)
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.empty_cache()
        self.device = torch.device("cuda:%d" % args.cuda if torch.cuda.is_available() else "cpu")
        gpu_brand = torch.cuda.get_device_name(args.cuda) if use_cuda else None
        gpu_count = torch.cuda.device_count() if args.use_multigpu else 1
        if use_cuda:
            print('Using %d CUDA cores [%s] for training!' % (gpu_count, gpu_brand))
        args_subset = ['exp', 'model', 'lr', 'batch_size']
        self.book = BookKeeper(args, args_subset)
        self.args = self.book.args
        self.batch_size = args.batch_size
        self.curriculum = args.curriculum
        self.dtype = torch.float32
        self.epochs_completed = self.book.last_epoch
        self.time = 15 
        self.model = args.model
        self.model_pose = LSTM_Action_Classifier().to(self.device).float()
        self.optimizer_model_pose = eval(args.optimizer)(self.model_pose.parameters(), lr = args.lr)
        self.scheduler_pose = eval(args.scheduler)(self.optimizer_model_pose, factor=args.factor, patience=args.patience, threshold= args.threshold,  min_lr = 2e-7)
       
        print('Model Created')
        if args.load:
            print('Loading Model')
            self.book._load_model(self.model_pose, 'model_pose')
           
        print('Loading the data')
        self.load_data(args)
        print('Data loading completed') 
        

    def load_data(self,args):
        ds_train = LoadData_GRAB(args, dataset_dir=args.dataset_dir, ds_name='train', load_on_ram=args.load_on_ram)
        self.ds_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
        print('Train set loaded. Size=', len(self.ds_train.dataset))
        ds_val = LoadData_GRAB(args, dataset_dir=args.dataset_dir, ds_name='val', load_on_ram=args.load_on_ram)
        self.ds_val = DataLoader(ds_val, batch_size=1, shuffle=True, num_workers=0, drop_last=True)
        print('Validation set loaded. Size=', len(self.ds_val.dataset))
            
    def load_batch(self, batch, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        fullpose_with_fingers =  batch['fullpose'][:, :self.time, ...].to(self.device).float()
        transl =  batch['transl'][:, :self.time, ...].to(self.device).float().reshape(batch_size, self.time, -1)
        sbj_joints = batch['joints'][:, :self.time, ...].reshape(-1, self.time, 127, 3).to(self.device).float()
        fullpose_rot =  fullpose_with_fingers[:, :, used_joints, :].to(self.device).float()
        right_arm_rot =  fullpose_with_fingers[:, :, right_arm_joints, :].to(self.device).float().reshape(batch_size, self.time, -1)
        left_arm_rot =  fullpose_with_fingers[:, :, left_arm_joints, :].to(self.device).float().reshape(batch_size, self.time, -1)
        arms_rot =  fullpose_with_fingers[:, :, arm_joints, :].to(self.device).float().reshape(batch_size, self.time, -1)
        right_hand_finger_rot =  fullpose_with_fingers[:, :, right_hand_finger_joints, :].to(self.device).float().reshape(batch_size, self.time, -1)
        lefthand_finger_rot =  fullpose_with_fingers[:, :, left_hand_finger_joints, :].to(self.device).float().reshape(batch_size, self.time, -1)
        upperbody_rot =  fullpose_with_fingers[:, :, upperbody_joints, :].to(self.device).float().reshape(batch_size, self.time, -1)
        lowerbody_rot =  fullpose_with_fingers[:, :, lowerbody_joints, :].to(self.device).float().reshape(batch_size, self.time, -1)
        betas = batch['betas'].squeeze(1).to(self.device).float()
        rhand_contact = batch['rhand_contact'].to(self.device).float()
        lhand_contact = batch['lhand_contact'].to(self.device).float()
        right_hand_vert = batch['rhand_verts'][:, :self.time, ...].to(self.device).float()
        left_hand_vert = batch['lhand_verts'][:, :self.time, ...].to(self.device).float()
        verts = batch['verts'][:, :self.time, ...].to(self.device).float()
        verts_obj = batch['verts_obj'][:, :self.time, ...].to(self.device).float()
        obj_global_orient = batch['global_orient_obj'][:, :self.time, ...].to(self.device).float().reshape(batch_size, self.time, -1)
        obj_transl = batch['transl_obj'][:, :self.time, ...].to(self.device).float()
        data = {
            'action_labels': [action_classes[batch['intent'][k]] for k in range(len(batch['intent']))],
            'joints': sbj_joints,
            'gender': batch['gender'] ,
            'sbj_v_template': batch['sbj_v_template'],
            'upperbody_rot': upperbody_rot,
            'right_arm_rot': right_arm_rot,
            'left_arm_rot': left_arm_rot,
            'arms_rot': arms_rot,
            'righthand_finger_rot': right_hand_finger_rot,
            'lefthand_finger_rot': lefthand_finger_rot,
            'lowerbody_rot': lowerbody_rot,
            'fullpose': fullpose_rot,
            'fullpose_with_fingers': fullpose_with_fingers,
            'transl': transl,
            'betas': betas,
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

    def train(self, num_epoch):
        running_loss = 0
        running_count = 1
        self.model_pose.train()
        T = self.time
        training_tqdm = tqdm(self.ds_train, desc='train' + ' {:.10f}'.format(0), leave=False, ncols=120)
        
        for count, batch in enumerate(training_tqdm):
            data = self.load_batch(batch)
            # with torch.autograd.detect_anomaly():
            if True:
                self.optimizer_model_pose.zero_grad()
                joints3d_vec = data['joints'][..., :22, :]
                loss_model, pred_label = self.model_pose(joints3d_vec, data['intent_vec'])                 
                running_count = (count+1) * self.time 
                running_loss += loss_model.item()
                
                training_tqdm.set_description('Train {:.8f} '.format(running_loss/running_count))
                training_tqdm.refresh()
                loss_model.backward()
                if loss_model == float('inf') or loss_model == nan:
                    print('Train loss is nan')
                    exit()
                torch.nn.utils.clip_grad_norm_(self.model_pose.parameters(), 1)
                self.optimizer_model_pose.step()
        return running_loss/running_count

    def evaluate(self, num_epoch=0):
        running_loss = 0
        running_count = 1
        self.model_pose.eval()
        T = self.time
        eval_tqdm = tqdm(self.ds_val, desc='eval' + ' {:.10f}'.format(0), leave=False, ncols=120)
        for count, batch in enumerate(eval_tqdm):
            data = self.load_batch(batch, batch_size=1)
            joints3d_vec = data['joints'][..., :22, :]
            loss_model, pred_label = self.model_pose(joints3d_vec, data['intent_vec'])                 
            running_count = (count+1) * self.time 
            running_loss += loss_model.item()
            eval_tqdm.set_description('Val {:.8f} '.format(running_loss/running_count))
            eval_tqdm.refresh()      
        return running_loss/running_count
    
    def fit(self, n_epochs=None, message=None):
        print('*****Inside Trainer.fit *****')
        if n_epochs is None:
            n_epochs = self.args.num_epochs
        
        starttime = datetime.now().replace(microsecond=0)
        print('Started Training at', datetime.strftime(starttime, '%Y-%m-%d_%H:%M:%S'), 'Total epochs: ', n_epochs)
        save_model_dict = {}
        
        self.mask = masking(self.time, 1)
        for epoch_num in range(self.epochs_completed, n_epochs + 1):
            tqdm.write('--- starting Epoch # %03d' % epoch_num)
            train_loss = self.train(epoch_num)
            eval_loss  = self.evaluate(epoch_num)
            self.scheduler_pose.step(train_loss)
            tqdm.write('Training up to frame: {}'.format(self.time))
            self.book.update_res({'epoch': epoch_num, 'train': train_loss, 'val': eval_loss, 'test': 0.0})
            self.book._save_res()
            self.book.print_res(epoch_num, key_order=['train', 'val', 'test'], lr=self.optimizer_model_pose.param_groups[0]['lr'])
            if epoch_num > 20 and epoch_num % 20 == 0:
                f = open(os.path.join(self.args.save_dir, self.book.name.name, self.book.name.name + '{:06d}'.format(epoch_num) + '.p'), 'wb') 
                save_model_dict.update({'model_pose': copy.deepcopy(self.model_pose.state_dict())})
                torch.save(save_model_dict, f)
                f.close()
        endtime = datetime.now().replace(microsecond=0)
        print('Finished Training at %s\n' % (datetime.strftime(endtime, '%Y-%m-%d_%H:%M:%S')))
        print('Training complete in %s!\n' % (endtime - starttime))


if __name__ == '__main__':
    args = argparseNloop()
    model_trainer = Trainer(args=args)
    print("** Method Inititalization Complete **")
    model_trainer.fit()
    #for evaluation
    # model_trainer.evaluate()