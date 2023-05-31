import os
from os.path import join
import numpy as np
import utils as ut
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# global parameters
config = dict()
config['n_epochs'] = 75
config['batch_size'] = 4
config['patch_shape'] = [64, 64, 64]

data_dir = '/swi-cnn-copy'
config['initial_learning_rate_d'] = [5e-3]
config['end_learning_rate_d'] = [5e-5]
config['initial_learning_rate_g'] = [1e-2,1e-3,1e-4,1e-5]
config['end_learning_rate_g'] = [1e-2,1e-3,1e-4,1e-5]
config['decay_steps'] = 25
model_dir =  join(data_dir, 'models')

subjects_list = '/training_subjects'
with open(subjects_list, 'r') as f:
    subjects = np.array([subject.strip() for subject in f.readlines()])

subjects_list_test = '/forrestgump_list'
with open(subjects_list_test, 'r') as f:
    subjects_test = np.array([subject.strip() for subject in f.readlines()])

config['training'] = subjects
config['test'] = subjects_test

config['discriminator_depth'] = 4
config['discriminator_batch_norm'] = True

# labels = {'rn_r': 1, 'rn_l': 2,
#                   'sn_r': 3, 'sn_l': 4,
#                   'den_r': 5, 'den_l': 6,
#                   'stn_r': 7, 'stn_l': 8}
labels = {'den_l': 6}

# lambda values, default labmda = 0.01
lambdas = [0.05]

# pre-trained iteration, default pre_trained_iter = 40
pre_trained_iter = [40]

# delayed updates for discriminator, default delayed_updates_d = 1
delayed_updates_d = [1]

# models
models = ['unet','unetpp']

for model in models:
    base_dir = '/models'
    for lr_g_index in range(len(config['initial_learning_rate_g'])):
        print("initial learning rate for generator: ", config['initial_learning_rate_g'][lr_g_index])
        for lr_d_index in range(len(config['initial_learning_rate_d'])):
            print("initial learning rate for discriminator: ", config['initial_learning_rate_d'][lr_d_index])
            if config['initial_learning_rate_g'][lr_g_index] == config['end_learning_rate_g'][lr_g_index]:
                exp_dir_ = join(base_dir, 'region', 'grid_experiment', model, '75', 'adam all',
                            'fixed lr_g=%.1e' % config['initial_learning_rate_g'][lr_g_index])
            else:
                exp_dir_ = join(base_dir, 'region', 'grid_experiment', model, '75', 'adam all', 
                            'lr_g=%.1e' % config['initial_learning_rate_g'][lr_g_index])
            if config['initial_learning_rate_d'][lr_d_index] == config['end_learning_rate_d'][lr_d_index]:
                exp_dir = join(exp_dir_, 'fixed lr_d=%.1e' % config['initial_learning_rate_d'][lr_d_index])
            else:
                exp_dir = join(exp_dir_, 'lr_d=%.1e' % config['initial_learning_rate_d'][lr_d_index])
            for lambda_value in lambdas:
                print("lambda: ", lambda_value)
                final_dir_l = join(exp_dir, 'lambda=%.3f' % lambda_value)
                for pre_trained_iter_value in pre_trained_iter:
                    print("pre trained: ", pre_trained_iter_value)
                    final_dir_ = join(final_dir_l, 'pre=%i' % pre_trained_iter_value)
                    for delayed_updates_d_value in delayed_updates_d:
                        print("delayed updates: ", delayed_updates_d_value)
                        final_dir = join(final_dir_, 'delayed_updates_d=%i' % delayed_updates_d_value)
                        for region in labels:
                            print("region: ", region, "label: ", labels[region])
                            config['batches_x_dir'] = join(data_dir, 'batches', 'region', 'unetpp',
                                            'mixed_sigmoid', region)
                            config['batches_y_dir'] = join(data_dir, 'batches', 'region', 'unetpp',
                                            'mixed_sigmoid', region)
                            model_path = join(final_dir, region)
                            if not os.path.exists(model_path):
                                print('Creating directory: {}'.format(model_path))
                                os.makedirs(model_path)
                            # train unet gan
                            ut.train_unet_gan(config = config,
                                            model_name = model,
                                            model_path = model_path,
                                            region = region,
                                            labels = labels[region],
                                            lambda_ = lambda_value,
                                            pretrained_iter = pre_trained_iter_value,
                                            lr_g_index = lr_g_index,
                                            lr_d_index = lr_d_index,
                                            delayed_updates_d = delayed_updates_d_value,
                            )
