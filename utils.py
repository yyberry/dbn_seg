from os.path import join
import ants
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR 
import torchvision.transforms as transforms
from metrics import dice_coeff, dice_loss
import models as mdl
from models import UNet3D, Discriminator
import matplotlib.pyplot as plt
from monai.networks.nets import BasicUNetPlusPlus
from monai.networks.layers.factories import Conv

import numpy as np
from sklearn.model_selection import KFold

import math
import os


def train_model(config, model_name, model_path, labels=None):
    """
    Train UNets with the given configuration.
    Parameters
    ----------
    config : dict
        Dictionary containing the configuration of the model.
    model_name : str
        Name of the model to be trained.
    model_path : str
        Path to the directory where the model will be saved.
    labels : int
    """
    # Define paths
    fmodel = join(model_path, 'checkpoint_final.pt')
    floss = join(model_path, 'loss.txt')
    floss_fig = join(model_path, 'loss.png')

    #define training steps
    n_steps = np.ceil(
        len(config['training']) / config['batch_size']).astype(int)
    kf = KFold(n_splits=n_steps)
    training_steps = [fold[1] for fold in kf.split(
                        range(len(config['training'])))]
    
    # load the test data randomly
    test_indices = np.random.choice(
                                        len(config['test']),
                                        config['batch_size'],
                                        replace=False
                                    )
    test_x, test_y = fetch_batch(config['batches_x_dir'],
                                        config['batches_y_dir'],
                                        config['test'][test_indices],
                                        labels=labels)
    # transform the data to tensor
    test_x = torch.from_numpy(test_x).float()
    test_y = torch.from_numpy(test_y).float()
    test_x, test_y = test_x.cuda(), test_y.cuda()
    if model_name == 'unet':
        model = UNet3D(in_ch=1, num_classes=1)
    elif model_name == 'unetpp':
        model = unetpp_generator()
    # model.summary()
    train_loss_list = []
    test_loss_list = []
    test_xepoch = []
    if torch.cuda.is_available():
        model = model.cuda()
        print('cuda available! Training initialized on gpu ...')
    elif not torch.cuda.is_available():
        print('cuda not available! Training initialized on cpu ...')

    # define the loss function
    criterion = dice_loss
    optimizer = Adam(model.parameters(), lr=config['initial_learning_rate'])

    #train the model
    for epoch in range(config['n_epochs']):
        train_loss = 0.0
        model.train()
        if config['end_learning_rate'] is not None:
            decayed_lr = decay_lr(config['initial_learning_rate'],
                                      config['end_learning_rate'],
                                      epoch,
                                      config['n_epochs'],
                                      decay_steps=config['decay_steps'])
            for params in optimizer.param_groups:                   
                params['lr'] = decayed_lr
        for step in range(n_steps):
            train_x, train_y = fetch_batch(
                    config['batches_x_dir'],
                    config['batches_y_dir'],
                    config['training'][training_steps[step]],
                    labels=labels,
                    n_epoch=epoch)
            # transform the data to tensor
            train_x = torch.from_numpy(train_x).float()
            train_y = torch.from_numpy(train_y).float()
            train_x, train_y = train_x.cuda(), train_y.cuda()
            optimizer.zero_grad()
            outputs = model(train_x)
            if model_name == 'unetpp':
                outputs = outputs[0]
            # print(outputs.shape, train_y.shape)
            loss = criterion(outputs, train_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            del train_x, train_y
            torch.cuda.empty_cache()
        train_loss_list.append(train_loss/n_steps)
        # test the model
        model.eval()
        # test the model for each 5 epochs
        if (epoch+1) % 5 == 0 or epoch == 0:
            test_xepoch.append(epoch+1)
            with torch.no_grad():
                target = model(test_x)
                if model_name == 'unetpp':
                    target = target[0]
                test_loss = criterion(target, test_y).item()
                test_loss_list.append(test_loss)

        #print epoch, loss, test_loss, decayed_lr        
        if (epoch+1) % 5 == 0 or epoch == 0:
            print('Epoch: {}/{}...'.format(epoch+1, config['n_epochs']),
                    'Training Loss: {:.6f}...'.format(train_loss/n_steps),
                    'Test Loss: {:.6f}...'.format(test_loss),
                    'Learning rate: {:.6f}'.format(decayed_lr))
            #plot the loss
            plot_train_test(x1=range(1,len(train_loss_list)+1), y1=train_loss_list,
                            x2=test_xepoch, y2=test_loss_list,
                            x_label='Epochs', y_label='Loss',
                            title='Loss: %f' % test_loss_list[-1], 
                            save_dir=floss_fig)
        else :
            print('Epoch: {}/{}...'.format(epoch+1, config['n_epochs']),
                    'Training Loss: {:.6f}...'.format(train_loss/n_steps),
                    'Learning rate: {:.6f}'.format(decayed_lr))
        # save the model for each 10 epochs
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), join(model_path, 'checkpoint_%d.pt' % (epoch+1)))
            
    #save the loss and metrics
    np.savez(floss, train_loss_list=train_loss_list, test_loss_list=test_loss_list)
    # save the model
    torch.save(model.state_dict(), fmodel)

## reference: Beliveau, V., Nørgaard, M., Birkl, C., Seppi, K., & Scherfler, C. (2021). Automated segmentation of deep brain nuclei using convolutional neural networks and susceptibility weighted imaging, Human Brain Mapping, Vol. 42, No. 15, pp. 4809-4822. https://doi.org/10.1002/hbm.25604
def decay_lr(initial_learning_rate,
             end_learning_rate,
             global_step,
             n_epochs,
             decay_steps=None):

    """ Step decay learning rate """

    if decay_steps is None:
        decay_rate = (end_learning_rate/initial_learning_rate)**(1/n_epochs)
        decayed_lr = initial_learning_rate*decay_rate**np.floor(global_step)
    else:
        total_steps = np.floor(float(n_epochs)/float(decay_steps)) - 1
        decay_rate = (end_learning_rate/initial_learning_rate)**(1/total_steps)
        decayed_lr = initial_learning_rate*decay_rate**np.floor(
                     float(global_step)/float(decay_steps))

    return decayed_lr 

def fetch_batch(x_dir, y_dir, subjects, labels=None, n_epoch=None):

    """ Fetch batch data """

    # if not isinstance(labels, list):
    #     labels_ = ut.concat_labels(labels, background=True)

    x = []
    y = []
    # print("Fetch batch data ...")
    # print(labels)
    # print labels
    # print("Lables number: ", labels)

    if n_epoch is None:

        for subject in subjects:
            # print(subject)
            fx = join(x_dir, subject, 'x.nii.gz')
            fy = join(y_dir, subject, 'y.nii.gz')

            x.append(np.expand_dims(ants.image_read(fx).numpy(), axis=0))
            if labels is None:
                y.append(np.expand_dims(ants.image_read(fy).numpy(), axis=0))
            elif isinstance(labels, int):
                y.append(np.expand_dims(
                    np.isin(ants.image_read(fy).numpy(), labels), axis=0))
            else:
                y.append(expand_labels(ants.image_read(fy).numpy(),
                                          labels=labels))

    else:

        for subject in subjects:

            fx = join(x_dir, subject, 'x_%i.nii.gz' % n_epoch)
            fy = join(y_dir, subject, 'y_%i.nii.gz' % n_epoch)

            x.append(np.expand_dims(ants.image_read(fx).numpy(), axis=0))
            if labels is None:
                y.append(np.expand_dims(ants.image_read(fy).numpy(), axis=0))
            elif isinstance(labels, int):
                y.append(np.expand_dims(
                    np.isin(ants.image_read(fy).numpy(), labels), axis=0))
            else:
                y.append(expand_labels(ants.image_read(fy).numpy(),
                                          labels=labels))

    return np.stack(x, axis=0), np.stack(y, axis=0)

def expand_labels(data, labels, n_labels=None, override_binary=None):

    """ Transform integer labels to binary expension """
    print("expand labels ...")

    if isinstance(labels, int):
        labels = [labels]

    if override_binary is not None and len(np.unique(data)) > 2:
        print(labels)
        raise ValueError('Expected binary label but found the above labels.')

    if n_labels is None:
        y = np.zeros([len(labels)] + list(data.shape), dtype=np.int8)
        for nl, label in enumerate(labels):
            print(nl, label)
            # original code
            y[nl, data == label] = 1
            # y[nl, data == labels[label]] = 1
    else:

        y = np.zeros([n_labels] + list(data.shape), dtype=np.int8)

        if override_binary is None:
            for label in labels:
                y[label, data == label] = 1
                # y[labels[label], data == labels[label]] = 1
        else:
            y[override_binary, data == 1] = 1

    return y


# plot the loss and dice coefficient for training and test set
def plot_train_test(x1, y1, x2, y2, x_label, y_label, title, save_dir):
    plt.figure()
    plt.plot(x1, y1)
    plt.plot(x2, y2)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(linestyle='--')
    plt.legend(['Training', 'Test'])
    plt.savefig(save_dir, format='png')
    plt.close()


def train_unet_gan(config, model_name, model_path, region, labels=None, 
                   lambda_ = 0.01, pretrained_iter = 0, 
                   lr_g_index = 0, lr_d_index = 0,
                   delayed_updates_d = 1):
    """
    Train the UNets GAN model

    Parameters
    ----------
    config : dict
        Configuration dictionary
    model_name : str
        Model name
    model_path : str
        Model path
    region : int
    lambda_ : float
        Weight of the adversarial loss
    pretrained_iter : int
        The number of epochs of the pretrained model
    lr_g_index : int
    lr_d_index : int
    delayed_updates_d : int
    """

    #define training steps
    n_steps = np.ceil(
        len(config['training']) / config['batch_size']).astype(int)
    kf = KFold(n_splits=n_steps)
    training_steps = [fold[1] for fold in kf.split(
                        range(len(config['training'])))]
    
    # load the test data randomly
    test_indices = np.random.choice(
                                        len(config['test']),
                                        config['batch_size'],
                                        replace=False
                                    )
    test_x, test_y = fetch_batch(config['batches_x_dir'],
                                        config['batches_y_dir'],
                                        config['test'][test_indices],
                                        labels=labels)
    # transform the data to tensor
    test_x = torch.from_numpy(test_x).float()
    test_y = torch.from_numpy(test_y).float()
    test_x, test_y = test_x.cuda(), test_y.cuda()
    # define the generator
    if model_name == 'unet':
        model = UNet3D(in_ch=1, num_classes=1)
    elif model_name == 'unetpp':
        model = unetpp_generator()
    # define the discriminator
    discriminator_output_kernel = \
                tuple(np.array(config['patch_shape']) // 2**(config['discriminator_depth']))
    discriminator = Discriminator(depth = config['discriminator_depth'],
                                  output_kernel=discriminator_output_kernel,
                                  batch_norm=config['discriminator_batch_norm'])
    
    loss_dict = {
        'g': [],
        'g_d': [],
        'g_dice': [],
        'd': [],
        'd_real': [],
        'd_fake': [],
        'dice_eval': []
    }
    dice_eval_x = []
    
    if torch.cuda.is_available():
        model = model.cuda()
        discriminator = discriminator.cuda()
        print('cuda available! Training initialized on gpu ...')
    elif not torch.cuda.is_available():
        print('cuda not available! Training initialized on cpu ...')
    
    # load the pretrained model
    if pretrained_iter > 0:
        base_dir = '/home/j/jh/175/q175jh/my_code/'
        if model_name == 'unet':
            pre_trained_model_path = join(base_dir, 'models_new', 'region', 'unet', 'final lr=0.005', region)
            model.load_state_dict(torch.load(join(pre_trained_model_path, 'checkpoint_%d.pt' % pretrained_iter)))
        elif model_name == 'unetpp':
            pre_trained_model_path = join(base_dir, 'models', 'region', 'unetpp', 'final lr=0.005', region)
            model.load_state_dict(torch.load(join(pre_trained_model_path, 'checkpoint_%d.pt' % pretrained_iter)))
        print('pretrained model loaded ...')

    # define the optimizer
    g_optimizer= optim.Adam(model.parameters(), lr=config['initial_learning_rate_g'][lr_g_index], betas=(0.5, 0.999))
    d_optimizer= optim.Adam(discriminator.parameters(), lr=config['initial_learning_rate_d'][lr_d_index], betas=(0.5, 0.999))

    # define the optimizer by sgd
    # g_optimizer= optim.SGD(model.parameters(), lr=config['initial_learning_rate_g'][lr_g_index], momentum=0.9)
    # d_optimizer= optim.SGD(discriminator.parameters(), lr=config['initial_learning_rate_d'][lr_d_index], momentum=0.9)

    #train the model
    for epoch in range(config['n_epochs']):
        model.train()
        if config['end_learning_rate_g'][lr_g_index] is not None:
            decayed_lr_g = decay_lr(config['initial_learning_rate_g'][lr_g_index],
                                      config['end_learning_rate_g'][lr_g_index],
                                      epoch,
                                      config['n_epochs'],
                                      decay_steps=config['decay_steps'])
            for params in g_optimizer.param_groups:                   
                params['lr'] = decayed_lr_g
        if config['end_learning_rate_d'][lr_d_index] is not None:
            decayed_lr_d = decay_lr(config['initial_learning_rate_d'][lr_d_index],
                                        config['end_learning_rate_d'][lr_d_index],
                                        epoch,
                                        config['n_epochs'],
                                        decay_steps=config['decay_steps'])
            for params in d_optimizer.param_groups:
                params['lr'] = decayed_lr_d
        d_loss = []
        d_real = []
        d_fake = []
        g_loss = []
        g_d_loss = []
        g_dice_loss = []
        for step in range(n_steps):
            train_x, train_y = fetch_batch(
                    config['batches_x_dir'],
                    config['batches_y_dir'],
                    config['training'][training_steps[step]],
                    labels=labels,
                    n_epoch=epoch)
            # transform the data to tensor
            train_x = torch.from_numpy(train_x).float()
            train_y = torch.from_numpy(train_y).float()
            # train the discriminator
            d_loss_iter = []
            d_real_iter = []
            d_fake_iter = []
            if epoch % delayed_updates_d == 0:
                # print('training discriminator ...')
                train_discriminator_iter(discriminator, model, model_name,
                                        d_optimizer, config, labels, 
                                        d_loss_iter, d_real_iter, d_fake_iter)
                d_loss.append(np.mean(d_loss_iter))
                d_real.append(np.mean(d_real_iter))
                d_fake.append(np.mean(d_fake_iter))
            else:
                d_loss.append(loss_dict['d'][-1])
                d_real.append(loss_dict['d_real'][-1])
                d_fake.append(loss_dict['d_fake'][-1])
            # print(len(d_loss_iter))
            # train the generator
            train_generator_iter(D = discriminator, G = model, G_name = model_name,
                                 train_x = train_x, train_y = train_y, 
                                 g_optimizer = g_optimizer, g_loss_iter = g_loss, 
                                 g_d_loss_iter = g_d_loss, g_dice_loss_iter = g_dice_loss, 
                                 lambda_adv = lambda_
                )
        
        # record the loss
        loss_dict['g'].append(np.mean(g_loss))
        loss_dict['g_d'].append(np.mean(g_d_loss))
        loss_dict['g_dice'].append(np.mean(g_dice_loss))
        loss_dict['d'].append(np.mean(d_loss))
        loss_dict['d_real'].append(np.mean(d_real))
        loss_dict['d_fake'].append(np.mean(d_fake))
        

        d_real_ = np.array(loss_dict['d_real'])
        d_fake_ = np.array(loss_dict['d_fake'])
        d_output_real = sigmoid_function(d_real_ - d_fake_)
        d_output_fake = sigmoid_function(d_fake_ - d_real_)
        if d_output_real[-1] < 0.6 and d_output_fake[-1] > 0.4:
            # save the model
            fout = join(model_path, 'checkpoint_%i.pt' % (epoch + 1))
            torch.save({
                            'n_train': epoch + 1,
                            'G': model.state_dict(),
                            'D': discriminator.state_dict(),
                            'G_opt': g_optimizer.state_dict(),
                            'D_opt': d_optimizer.state_dict(),
                            'loss_dict': loss_dict,
                        }, fout)
        model.eval()
        # test the model for each 5 epochs
        if (epoch+1) % 5 == 0 or epoch == 0:
            dice_eval_x.append(epoch+1)
            with torch.no_grad():
                target = model(test_x)
                if model_name == 'unetpp':
                    target = target[0]
                criterion = dice_loss
                test_loss = criterion(target, test_y).item()
                loss_dict['dice_eval'].append(test_loss)
            # print(dice_eval_x)
            # print('Epoch %i: dice_eval: %f' % (epoch+1, test_loss))
            # plot the loss
            plot_loss_dict(loss_dict, dice_eval_x, model_path, config, lr_g_index, lr_d_index)
            # plot loss for generator and discriminator
            plot_loss_g_d(loss_dict, model_path, config, lr_g_index, lr_d_index)
            # plot outputs from discriminator
            plot_cx(loss_dict, model_path, config, lr_g_index, lr_d_index)
            plot_d_outputs(loss_dict, model_path, config, lr_g_index, lr_d_index)

        #print the loss
        loss_str = 'Epoch %i: ' % (epoch+1)
        loss_str += ', '.join([key + ': %f' % loss_dict[key][-1]
                               for key in loss_dict.keys()])
        print(loss_str)
        
        # save the model
        if (epoch + 1) % 10 == 0:
                    fout = join(model_path, 'checkpoint_%i.pt' % (epoch + 1))
                    torch.save({
                            'n_train': epoch + 1,
                            'G': model.state_dict(),
                            'D': discriminator.state_dict(),
                            'G_opt': g_optimizer.state_dict(),
                            'D_opt': d_optimizer.state_dict(),
                            'loss_dict': loss_dict,
                        }, fout)
        # erly stopping when the D(real) is close to 1 and D(fake) is close to 0 for 10 epochs
        if epoch >= 25:
            counter = 0
            for i in range(1, 11):
                if d_output_real[-i] > 0.99 and d_output_fake[-i] < 0.01:
                    counter += 1
                    # print counter/10
                    print("counter: %i / 10" % counter)
                else:
                    break
            if counter == 10:
                print('Early stopping')
                break
    # save the final model
    fout = join(model_path, 'checkpoint_final.pt')
    torch.save({
            'n_train': epoch + 1,
            'G': model.state_dict(),
            'D': discriminator.state_dict(),
            'G_opt': g_optimizer.state_dict(),
            'D_opt': d_optimizer.state_dict(),
            'loss_dict': loss_dict,
        }, fout)

# train the discriminator
def train_discriminator_iter(D, G, G_name, d_optimizer, config, labels, d_loss_iter, d_real_iter, d_fake_iter):
    # Forward propagation: Turn on automatic derivation anomaly detection
    # torch.autograd.set_detect_anomaly(True)

    D.requires_grad = True
    G.eval()

    # load a random batch
    indices = np.random.choice(
                                len(config['training']),
                                config['batch_size'],
                                replace=False
                               )
    train_x, train_y = fetch_batch(config['batches_x_dir'],
                                        config['batches_y_dir'],
                                        config['training'][indices],
                                        labels=labels)
    
    # transform the data to tensor
    train_x = torch.from_numpy(train_x).float()
    train_y = torch.from_numpy(train_y).float()
    train_x, train_y = train_x.cuda(), train_y.cuda()

    # generate fake data
    fake_y = G(train_x)

    if G_name == 'unetpp':
        fake_y = fake_y[0]

    # Calulate discriminator outputs
    d_real = D(train_y)
    d_fake = D(fake_y)

    d_optimizer.zero_grad()

    # Calculate discriminator loss
    d_loss = -torch.log(torch.sigmoid(d_real - d_fake)).mean()
    # print(d_loss.isnan().any())

    # with torch.autograd.detect_anomaly():
    #     d_loss.backward()
    d_loss.backward()
    d_optimizer.step()

    d_loss_iter.append(d_loss.detach().cpu().numpy())
    d_real_iter.append(d_real.mean().detach().cpu().numpy())
    d_fake_iter.append(d_fake.mean().detach().cpu().numpy())

    del train_x, train_y
    torch.cuda.empty_cache()

# train the generator
def train_generator_iter(D, G, G_name, train_x, train_y, g_optimizer, g_loss_iter, g_d_loss_iter, g_dice_loss_iter, lambda_adv = 0.01):
    G.train()
    train_x, train_y = train_x.cuda(), train_y.cuda()

    # generate fake data
    fake_y = G(train_x)

    if G_name == 'unetpp':
        fake_y = fake_y[0]

    # fix the discriminator
    D.requires_grad = False

    # Evaluate discriminator on fake data
    d_fake = D(fake_y)
    d_real = D(train_y)

    # reset dradients
    g_optimizer.zero_grad()

    # caclculate adverssarial loss
    g_d_loss = -torch.log(torch.sigmoid(d_fake - d_real)).mean()

    # calculate dice loss
    dice = dice_loss(fake_y, train_y)

    # total loss with dice loss
    g_loss = (1-lambda_adv) * dice + lambda_adv * g_d_loss

    g_loss.backward()
    g_optimizer.step()

    g_loss_iter.append(g_loss.detach().cpu().numpy())
    g_d_loss_iter.append(g_d_loss.detach().cpu().numpy())
    g_dice_loss_iter.append(dice.detach().cpu().numpy())

    del train_x, train_y
    torch.cuda.empty_cache()

def plot_loss_dict(loss_dict, dice_eval_x, out_dir, config, lr_g_index, lr_d_index):
    for loss in loss_dict.keys():
        # plot the test loss for each five epochs
        if loss == 'dice_eval':
            plt.plot(dice_eval_x, loss_dict[loss], label = loss)
        else:
            plt.plot(loss_dict[loss], label = loss)
        # title with lr_g, lr_d
        plt.title('lr_g: %.1e ~ %.1e, \n lr_d: %.1e ~ %.1fe' % (config['initial_learning_rate_g'][lr_g_index], config['end_learning_rate_g'][lr_g_index],
                                                    config['initial_learning_rate_d'][lr_d_index], config['end_learning_rate_d'][lr_d_index]))
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)
        plt.legend()
        plt.ylim(0., 1.)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(linestyle='--')
        fout = join(out_dir, loss + '.png')
        plt.savefig(fout, format='png')
        plt.close()

def plot_loss_g_d(loss_dict, out_dir, config, lr_g_index, lr_d_index):
    plt.figure()
    plt.plot(loss_dict['g'], label='g')
    plt.plot(loss_dict['d'], label='d')
    plt.title('lr_g: %.1e ~ %.1e, \n lr_d: %.1e ~ %.1e' % (config['initial_learning_rate_g'][lr_g_index], config['end_learning_rate_g'][lr_g_index],
                                                config['initial_learning_rate_d'][lr_d_index], config['end_learning_rate_d'][lr_d_index]))
    # plt.plot(sigmoid_function(d_real - d_fake), label='d_real')
    # plt.plot(sigmoid_function(d_fake - d_real), label='d_fake')
    plt.legend()
    plt.ylim(0., 1.)
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.grid(linestyle='--')
    plt.savefig(join(out_dir, 'loss.png'))
    plt.close()

def plot_cx(loss_dict, out_dir, config, lr_g_index, lr_d_index):
    plt.figure()
    plt.plot(loss_dict['d_real'], label='d_real')
    plt.plot(loss_dict['d_fake'], label='d_fake')
    plt.title('lr_g: %.1e ~ %.1e, \n lr_d: %.1e ~ %.1e' % (config['initial_learning_rate_g'][lr_g_index], config['end_learning_rate_g'][lr_g_index],
                                                config['initial_learning_rate_d'][lr_d_index], config['end_learning_rate_d'][lr_d_index]))
    # plt.plot(sigmoid_function(d_real - d_fake), label='d_real')
    # plt.plot(sigmoid_function(d_fake - d_real), label='d_fake')
    plt.legend()
    plt.ylim(0., 1.)
    plt.xlabel('Epochs')
    plt.ylabel('C(x)')
    plt.grid(linestyle='--')
    plt.savefig(join(out_dir, 'C(x).png'))
    plt.close()

def sigmoid_function(z):
    fz = []
    for num in z:
        fz.append(1/(1 + math.exp(-num)))
    return fz

def plot_d_outputs(loss_dict, out_dir, config, lr_g_index, lr_d_index):
    # d_real and d_fake to numpy 
    d_real = np.array(loss_dict['d_real'])
    d_fake = np.array(loss_dict['d_fake'])

    plt.figure()
    plt.plot(sigmoid_function(d_real - d_fake), label='d_real')
    plt.plot(sigmoid_function(d_fake - d_real), label='d_fake')
    plt.title('lr_g: %.1e ~ %.1e, \n lr_d: %.1e ~ %.1e' % (config['initial_learning_rate_g'][lr_g_index], config['end_learning_rate_g'][lr_g_index],
                                                config['initial_learning_rate_d'][lr_d_index], config['end_learning_rate_d'][lr_d_index]))
    plt.legend()
    plt.ylim(-0.1, 1.1)
    plt.xlabel('Epochs')
    plt.ylabel('D(x)')
    plt.grid(linestyle='--')
    plt.savefig(join(out_dir, 'D(x).png'))
    plt.close()

# test the discriminator to output its architecture
def test_discriminator():
    # define the discriminator
    discriminator_output_kernel = \
                tuple(np.array([64, 64, 64]) // 2**(4))
    D = Discriminator(depth = 4, output_kernel=discriminator_output_kernel, 
                      batch_norm=True)
    D.cuda()
    print(D)

# test the unet++ to output its architecture
def unetpp_generator():
    # x = torch.randn(2,10).cuda()
    model = BasicUNetPlusPlus(
                                    spatial_dims=3,
                                    in_channels=1,
                                    out_channels=1,
                                    features=(16, 32, 64, 128, 256, 16)
                                )
    conv = model.final_conv_0_4
    new_conv = nn.Sequential(
        conv,
        nn.Sigmoid()
    )
    model.final_conv_0_4 = new_conv
    return model

def load_optimizer_get_lr():
    # model is saved as following
    # torch.save({
    #         'n_train': epoch + 1,
    #         'G': model.state_dict(),
    #         'D': discriminator.state_dict(),
    #         'G_opt': g_optimizer.state_dict(),
    #         'D_opt': d_optimizer.state_dict(),
    #         'loss_dict': loss_dict,
    #     }, fout)
    base_dir = '/home/j/jh/175/q175jh/my_code/'
    final_dir = join(base_dir, 'models', 'region', 'unet_gan', 'binarized change lr', 'fixed lr_g=0.1000 new',
                         'lr_d=0.0050', 'lambda=0.050', 'pre=40', 'den_l')
    discriminator_output_kernel = \
                tuple(np.array([64, 64, 64]) // 2**(4))
    discriminator = Discriminator(depth = 4,
                                  output_kernel=discriminator_output_kernel,
                                  batch_norm=True)
    # optimizer Adam
    d_optimizer= optim.Adam(discriminator.parameters(), lr=1, betas=(0.5, 0.999))
    # load the optimizer
    d_optimizer.load_state_dict(torch.load(join(final_dir, 'checkpoint_30.pt'))['D_opt'])
    # print the learning rate
    print(d_optimizer.param_groups[0]['lr'])
    final_dir_ = join(base_dir, 'models', 'region', 'compare_gan', 'unetpp', 'adam all', 'fixed lr_g=0.1000',
                         'lr_d=0.0050', 'lambda=0.050', 'pre=40', 'den_l')
    # optimizer Adam
    d_optimizer_= optim.Adam(discriminator.parameters(), lr=1, betas=(0.5, 0.999))
    # load the optimizer
    d_optimizer_.load_state_dict(torch.load(join(final_dir_, 'checkpoint_30.pt'))['D_opt'])
    # print the learning rate
    print(d_optimizer_.param_groups[0]['lr'])
    

def load_loss_dict():
    # model is saved as following
    # torch.save({
    #         'n_train': epoch + 1,
    #         'G': model.state_dict(),
    #         'D': discriminator.state_dict(),
    #         'G_opt': g_optimizer.state_dict(),
    #         'D_opt': d_optimizer.state_dict(),
    #         'loss_dict': loss_dict,
    #     }, fout)
    config = dict()
    config['initial_learning_rate_d'] = [1e-5]
    config['end_learning_rate_d'] = [1e-5]
    config['initial_learning_rate_g'] = [5e-2]
    config['end_learning_rate_g'] = [5e-2]
    base_dir = '/home/j/jh/175/q175jh/my_code/'
    final_dir = join(base_dir, 'models', 'region', 'gan_lessd_5', 'unetpp', '75', 'adam all','fixed lr_g=5.0e-02',
                         'fixed lr_d=1.0e-05', 'lambda=0.050', 'pre=40', 'delayed_updates_d=2', 'den_l')
    # load the loss_dict
    loss_dict = {
        'g': [],
        'g_d': [],
        'g_dice': [],
        'd': [],
        'd_real': [],
        'd_fake': [],
        'dice_eval': []
    }
    loss_dict = torch.load(join(final_dir, 'checkpoint_final.pt'), map_location=torch.device('cpu'))['loss_dict']
    # plot the loss by using plot_d_outputs
    out_dir = join(base_dir, 'labels', 'reproduced_figures', 'gan_lessd_5', 'fixed lr_d=%.1e' % config['initial_learning_rate_d'][0],
                   'delayed_updates_d=2')
    if not os.path.exists(out_dir):
        print('Creating directory: {}'.format(out_dir))
        os.makedirs(out_dir)
    plot_d_outputs(loss_dict, out_dir, config, 0, 0)


