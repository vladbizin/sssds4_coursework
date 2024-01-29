import os
import numpy as np
import torch
import torch.nn as nn

from util import find_max_epoch, print_size, training_loss, calc_diffusion_hyperparams
from util import get_mask_mnr, get_mask_bm, get_mask_rm
from util import TS_Dataset

from torch.utils.data import DataLoader
from imputers.SSSDS4Imputer import SSSDS4Imputer


def train(output_directory,
          ckpt_iter,
          data_path,
          n_iters,
          iters_per_ckpt,
          iters_per_logging,
          learning_rate,
          only_generate_missing,
          masking,
          missing_k,
          batch_size,
          diffusion_config,
          model_config
          ):
    
    """
    Train Diffusion Models

    Parameters:
    output_directory (str):         save model checkpoints to this path
    ckpt_iter (int or 'max'):       the pretrained checkpoint to be loaded; 
                                    automatically selects the maximum iteration if 'max' is selected
    data_path (str):                path to dataset, numpy array.
    n_iters (int):                  number of iterations to train
    iters_per_ckpt (int):           number of iterations to save checkpoint, 
                                    default is 10k, for models with residual_channel=64 this number can be larger
    iters_per_logging (int):        number of iterations to save training log and compute validation loss, default is 100
    learning_rate (float):          learning rate

    use_model (int):                0:DiffWave. 1:SSSDSA. 2:SSSDS4.
    only_generate_missing (int):    0:all sample diffusion.  1:only apply diffusion to missing portions of the signal
    masking(str):                   'mnr': missing not at random, 'bm': blackout missing, 'rm': random missing
    missing_k (int):                k missing time steps for each feature across the sample length.
    batch_size (int):               number of samples in each batch.
    """

    
    # generate experiment (local) path
    local_path = "T{}_beta0{}_betaT{}".format(diffusion_config["T"],
                                              diffusion_config["beta_0"],
                                              diffusion_config["beta_T"])
    
    # calculate diffusion hyperparams
    diffusion_hyperparams = calc_diffusion_hyperparams(
        **diffusion_config)
    
    # Get shared output_directory ready
    output_directory = os.path.join(output_directory, local_path)
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    print("output directory", output_directory, flush=True)

    # map diffusion hyperparameters to gpu
    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()

    # predefine model
    net = SSSDS4Imputer(**model_config).cuda()

    # print information about model
    print_size(net)

    # define optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # load checkpoint
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(output_directory)
    if ckpt_iter >= 0:
        try:
            # load checkpoint file
            model_path = os.path.join(output_directory, '{}.pkl'.format(ckpt_iter))
            checkpoint = torch.load(model_path, map_location='cpu')

            # feed model dict and optimizer state
            net.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            print('Successfully loaded model at iteration {}'.format(ckpt_iter))
        except:
            ckpt_iter = -1
            print('No valid checkpoint model found, start training from initialization try.')
    else:
        ckpt_iter = -1
        print('No valid checkpoint model found, start training from initialization.')

        
        
    
    ### Custom data loading and reshaping ###
        

    train_ds = TS_Dataset(data_path, "cuda")
    train_loader = DataLoader(train_ds, shuffle=True, batch_size=batch_size)

    # training_data = np.load(data_path)
    # num_batches = training_data.shape[0]//batch_size
    # len = training_data.shape[0]//num_batches * num_batches
    # training_data = training_data[:len]
    # training_data = np.split(training_data, num_batches, 0)
    # training_data = np.array(training_data)
    # training_data = torch.from_numpy(training_data).float().cuda()
    # print('Data loaded')

    
    
    # training
    n_iter = ckpt_iter + 1
    while n_iter < n_iters + 1:
        for batch in train_loader:

            if masking == 'rm':
                transposed_mask = get_mask_rm(batch[0], missing_k)
            elif masking == 'mnr':
                transposed_mask = get_mask_mnr(batch[0], missing_k)
            elif masking == 'bm':
                transposed_mask = get_mask_bm(batch[0], missing_k)

            mask = transposed_mask.permute(1, 0)
            mask = mask.repeat(batch.size()[0], 1, 1).float().cuda()
            loss_mask = ~mask.bool()
            batch = batch.permute(0, 2, 1)

            assert batch.size() == mask.size() == loss_mask.size()

            # back-propagation
            optimizer.zero_grad()
            X = batch, batch, mask, loss_mask
            loss = training_loss(net, nn.MSELoss(), X, diffusion_hyperparams,
                                 only_generate_missing=only_generate_missing)

            loss.backward()
            optimizer.step()

            if n_iter % iters_per_logging == 0:
                print("iteration: {} \tloss: {}".format(n_iter, loss.item()))

            # save checkpoint
            if n_iter > 0 and n_iter % iters_per_ckpt == 0:
                checkpoint_name = '{}.pkl'.format(n_iter)
                torch.save({'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                           os.path.join(output_directory, checkpoint_name))
                print('model at iteration %s is saved' % n_iter)

            n_iter += 1
