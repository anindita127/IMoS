import argparse
import itertools
import sys
import os
from ast import literal_eval

def get_args_update_dict(args):
    args_update_dict = {}
    for string in sys.argv:
        string = ''.join(string.split('-'))
        if string in args:
            args_update_dict.update({string: args.__dict__[string]})
    return args_update_dict


def argparseNloop():
    parser = argparse.ArgumentParser()
    base_path = ''

    '''Directories and data path'''
    parser.add_argument('--work-dir', default = 'src', type=str,
                        help='The path to the downloaded grab data')
    parser.add_argument('--data-path', default = os.path.join(base_path, '..', 'DATASETS', 'GRAB'), type=str,
                        help='The path to the folder that contains the downloaded GRAB dataset')
    parser.add_argument('--model_path', default = 'smplx_model', type=str,
                        help='The path to the folder containing SMPLX model')
    parser.add_argument('--out-path', default = 'save', type=str,
                        help='The path to the folder to save the processed data')
    parser.add_argument('--render-path', default = 'render', type=str,
                        help='The path to the folder to save the rendered output')
    parser.add_argument('--data_dir', default = 'post_processed', type=str,
                        help='The path to the pre-processed data')
    
    
    '''Dataset Parameters'''
    parser.add_argument('-dataset', nargs='+', type=str, default='GRAB',
                        help='name of the dataset')
    parser.add_argument('-language_model', nargs='+', type=str, default='clip',
                        help='language model used')
    parser.add_argument('--max_frames', nargs='+', type=int, default=20,
                        help='Number of frames taken from each sequence in the dataset for training.')
    parser.add_argument('-seedLength', nargs='+', type=int, default=20,
                        help='initial length of inputs to seed the prediction; used when offset > 0')
    parser.add_argument('-exp', nargs='+', type=int, default=0,
                        help='experiment number')
    parser.add_argument('-framerate', nargs='+', type=int, default=30,  #this is a dummy value, not used anywhere.
                        help='frame rate after pre-processing.')
    parser.add_argument('-seed', nargs='+', type=int, default=4815,
                        help='manual seed')
    parser.add_argument('-load', nargs='+', type=str, default=None,
                        help='Load weights from this file')
    parser.add_argument('-cuda', nargs='+', type=int, default=0,
                        help='choice of gpu device, -1 for cpu')
    parser.add_argument('-overfit', nargs='+', type=int, default=0,
                        help='disables early stopping and saves models even if the dev loss increases. useful for performing an overfitting check')
    parser.add_argument('--norm_axis_angle', nargs='+', type=int, default=0,
                        help='if 1, then used pre-processed data with normalized axis angle.')

    '''Training parameters'''
    parser.add_argument('-model', nargs='+', type=str, default='CVAE_object_nojoint',
                        help='name of model')
    parser.add_argument('--use_discriminator', default=False,
                        help='Train the model with a discriminator')
    parser.add_argument('-batch_size', nargs='+', type=int, default=64,
                        help='minibatch size.')
    parser.add_argument('-num_epochs', nargs='+', type=int, default=1500,
                        help='number of epochs for training')
    parser.add_argument('--latentD', nargs='+', type=int, default=32,   #use 32 for training arms and 100 for training body
                        help='latent dimension of manifold space')
    parser.add_argument('--skip_train', nargs='+', type=int, default=1,
                        help='downsampling factor of the training dataset. For example, a value of s indicates floor(D/s) training samples are loaded, '
                        'where D is the total number of training samples (default: 1).')
    parser.add_argument('--skip_val', nargs='+', type=int, default=1,
                        help='downsampling factor of the validation dataset. For example, a value of s indicates floor(D/s) validation samples are loaded, '
                        'where D is the total number of validation samples (default: 1).')
    parser.add_argument('-early_stopping', nargs='+', type=int, default=0,
                        help='Use 1 for early stopping')
    parser.add_argument('--n_workers', default=0, type=int,
                        help='Number of PyTorch dataloader workers')
    parser.add_argument('-greedy_save', nargs='+', type=int, default=1,
                        help='save weights after each epoch if 1')
    parser.add_argument('-save_model', nargs='+', type=int, default=1,
                        help='flag to save model at every step')
    parser.add_argument('-stop_thresh', nargs='+', type=int, default=3,
                        help='number of consequetive validation loss increses before stopping')
    parser.add_argument('-eps', nargs='+', type=float, default=0,
                        help='if the decrease in validation is less than eps, it counts for one step in stop_thresh ')
    parser.add_argument('--curriculum', nargs='+', type=int, default=0,
                        help='if 1, learn generating time steps by starting with 2 timesteps upto time, increasing by a power of 2')
    parser.add_argument('--use-multigpu', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='If to use multiple GPUs for training')
    parser.add_argument('--load-on-ram', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='This will load all the data on the RAM memory for faster training.'
                             'If your RAM capacity is more than 40 Gb, consider using this.')
    
    '''Optimizer parameters'''
    parser.add_argument('--optimizer', default='optim.Adam', type=str,
                        help='Optimizer')
    parser.add_argument('-momentum', default=0.9, type=float,
                        help='Weight decay for SGD Optimizer')
    parser.add_argument('-lr', nargs='+', type=float, default=5e-4,
                        help='learning rate')
    
    '''Scheduler parameters'''
    parser.add_argument('--scheduler', default='torch.optim.lr_scheduler.ReduceLROnPlateau', type=str,
                        help='Scheduler')
    parser.add_argument('--patience', default=3, type=float,
                        help='Step size for ReduceOnPlateau scheduler')
    parser.add_argument('--factor', default=0.999, type=float,
                        help='Decay rate for ReduceOnPlateau scheduler')
    parser.add_argument('--threshold', default=0.05, type=float,
                        help='THreshold for ReduceOnPlateau scheduler')
    
    parser.add_argument('--stepsize', default=15, type=float,
                        help='Step size for StepLR scheduler')
    parser.add_argument('--gamma', default=0.99, type=float,
                        help='Decay rate for StepLR scheduler')
    '''Loss parameters'''
    parser.add_argument('--loss_kldiv', type=float, default=1e-3, 
                        help='weight of KL div loss')
    parser.add_argument('--loss_manifold', type=float, default=0.0, 
                        help='weight of manifold loss')
    parser.add_argument('--loss_gan', type=float, default=0.0, 
                        help='weight of GAN loss')
    parser.add_argument('--loss_contact', type=float, default=[0, 0], 
                        help='weight of contact loss')
    parser.add_argument('--loss_verts', type=float, default=0, 
                        help='weight of vertex loss')
    parser.add_argument('--loss_dist', type=float, default=1, 
                        help='weight of distance loss')
    parser.add_argument('--loss_reconstruction', type=float, default=1.0, 
                        help='weight of reconstruction and velocity loss')  
    parser.add_argument('--loss_angle_prior', type=float, default=0.0, 
                        help='weight for angle prior loss')                   


    args, unknown = parser.parse_known_args()
    if args.dataset.lower() == 'grab':
        args.num_pose_joints = 55
        args.num_pose_features = 6
        args.dataset_dir =  os.path.join('data', args.data_dir)
        args.save_dir =  os.path.join('save', args.data_dir)
    else:
        args.num_pose_joints = None
        args.num_pose_features = None
    if args.language_model.lower() == 'clip':
        args.text_embedding_dim = 512
    elif args.language_model.lower() == 'bert':
        args.text_embedding_dim = 768
    
    # print(args)
    return args
