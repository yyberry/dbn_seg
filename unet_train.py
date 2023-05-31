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
config['initial_learning_rate'] = 5e-3
config['end_learning_rate'] = 5e-5
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

labels = {'rn_r': 1, 'rn_l': 2,
                  'sn_r': 3, 'sn_l': 4,
                  'den_r': 5, 'den_l': 6,
                  'stn_r': 7, 'stn_l': 8}

# models
models = ['unetpp']

for model in models:
    base_dir = '/models'
    final_dir = join(base_dir, 'region', model, 'final lr=%.3f' % config['initial_learning_rate'])
    for region in labels:
        config['batches_x_dir'] = join(data_dir, 'batches', 'region', 'unetpp',
                        'mixed_sigmoid', region)
        config['batches_y_dir'] = join(data_dir, 'batches', 'region', 'unetpp',
                        'mixed_sigmoid', region)
        model_path = join(final_dir, region)
        if not os.path.exists(model_path):
            print('Creating directory: {}'.format(model_path))
            os.makedirs(model_path)
        ut.train_model(config = config, 
                       model_name = model, 
                       model_path = model_path, 
                       labels = labels[region])

    