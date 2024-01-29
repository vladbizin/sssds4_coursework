import os
import pickle
import torch
import torch.nn as nn
import numpy as np

from util import (
    find_max_epoch, print_size,
    training_loss, sampling,
    calc_diffusion_hyperparams
    )
from sklearn.model_selection import train_test_split

from util import TS_Dataset
from torch.utils.data import DataLoader

from imputers.SSSDS4Imputer import SSSDS4Imputer

from decimal import Decimal
from tqdm import tqdm




class SSSDS4():
    def __init__(
            self,
            K, L,
            T=200,
            beta_0=0.0001,
            beta_T=0.02,
            num_res_layers=36,
            res_channels=256, 
            skip_channels=256,
            diffusion_step_embed_dim_in=128,
            diffusion_step_embed_dim_mid=512,
            diffusion_step_embed_dim_out=512,
            s4_d_state=64,
            s4_dropout=0.0,
            s4_bidirectional=1,
            s4_layernorm=1
    ):
        
        # save diffustion config
        self._set_diffusion_config(
            T, beta_0, beta_T
        )

        # save model config
        self._set_model_config(
            K, K,
            num_res_layers,
            res_channels, 
            skip_channels,
            diffusion_step_embed_dim_in,
            diffusion_step_embed_dim_mid,
            diffusion_step_embed_dim_out,
            L,
            s4_d_state,
            s4_dropout,
            s4_bidirectional,
            s4_layernorm
        )

        # save params
        self.params={
            "diffusion_config": self.diffusion_config,
            "model_config": self.model_config
        }

        self.ckpt_epoch = -1


    def _set_diffusion_config(self, T, beta_0, beta_T):
        self.diffusion_config={
            "T": T,
            "beta_0": beta_0,
            "beta_T": beta_T
        }
        

    def _set_model_config(
            self,
            in_channels,
            out_channels,
            num_res_layers,
            res_channels, 
            skip_channels,
            diffusion_step_embed_dim_in,
            diffusion_step_embed_dim_mid,
            diffusion_step_embed_dim_out,
            s4_lmax,
            s4_d_state,
            s4_dropout,
            s4_bidirectional,
            s4_layernorm
    ):
        self.model_config={
            "in_channels": in_channels, 
            "out_channels": out_channels,
            "num_res_layers": num_res_layers,
            "res_channels": res_channels, 
            "skip_channels": skip_channels,
            "diffusion_step_embed_dim_in": diffusion_step_embed_dim_in,
            "diffusion_step_embed_dim_mid": diffusion_step_embed_dim_mid,
            "diffusion_step_embed_dim_out": diffusion_step_embed_dim_out,
            "s4_lmax": s4_lmax,
            "s4_d_state":s4_d_state,
            "s4_dropout":s4_dropout,
            "s4_bidirectional":s4_bidirectional,
            "s4_layernorm":s4_layernorm
        }


    def set_train_config(
                self,
                output_directory,
                epochs = 200,
                epochs_per_ckpt = 50,
                val_size = 0.1,
                learning_rate = 2e-4,
                only_generate_missing = 1,
                missing_mode = "rm",
                missing_r = "rand",
                batch_size = 8,
                verbose = 1,
        ):
            self.train_config={
                "output_directory": output_directory,
                "epochs": epochs,
                "epochs_per_ckpt": epochs_per_ckpt,
                "val_size": val_size,
                "learning_rate": learning_rate,
                "only_generate_missing": only_generate_missing,
                "missing_mode": missing_mode,
                "missing_r": missing_r,
                "batch_size": batch_size,
                "verbose": verbose
            }
            self.params["train_config"] = self.train_config

            # generate experiment (local) path
            local_path = "T{}_beta0{}_betaT{}".format(self.diffusion_config["T"],
                                                      self.diffusion_config["beta_0"],
                                                      self.diffusion_config["beta_T"])
            
            
            # get shared output_directory ready
            self.output_directory = os.path.join(output_directory, local_path)
            if not os.path.isdir(self.output_directory):
                os.makedirs(self.output_directory)
                os.chmod(self.output_directory, 0o775)
            print("Output directory: ", self.output_directory, flush=True)

            # save configs
            params_path = os.path.join(self.output_directory, 'params.pkl')
            with open(params_path, 'wb') as handle:
                pickle.dump(self.params, handle)

            # predefine model
            self.net = SSSDS4Imputer(**self.model_config).cuda()
            if self.train_config["verbose"]:
                print_size(self.net)

            self.optimizer = torch.optim.Adam(
                self.net.parameters(),
                lr=self.train_config["learning_rate"]
            )

            self.loss = nn.MSELoss()

            # calculate diffusion hyperparams
            self.diffusion_hyperparams = calc_diffusion_hyperparams(
                **self.diffusion_config)
            
            # map diffusion hyperparameters to gpu
            for key in self.diffusion_hyperparams:
                if key != "T":
                    self.diffusion_hyperparams[key] = self.diffusion_hyperparams[key].cuda()
    

    def load_state(
            self,
            output_directory,
            ckpt_epoch=-1
        ):

            self.ckpt_epoch = ckpt_epoch
            self.output_directory = output_directory

            
            # load model and train configs
            try:
                params_path = os.path.join(output_directory, 'params.pkl')
                with open(params_path, 'rb') as handle:
                    self.params = pickle.load(handle)
                self._set_diffusion_config(**self.params["diffusion_config"])
                self._set_model_config(**self.params["model_config"])
                self.set_train_config(**self.params["train_config"])
                print('Parameters loaded')
            except:
                    print('No saved parameters found, set configs and start training.')
                    return

            # initialize model and optimizer
            self.net = SSSDS4Imputer(**self.model_config).cuda()
            print_size(self.net)

            self.optimizer = torch.optim.Adam(
                self.net.parameters(),
                lr=self.train_config["learning_rate"]
            )

            self.loss = nn.MSELoss()

            # calculate diffusion hyperparams
            self.diffusion_hyperparams = calc_diffusion_hyperparams(
                **self.diffusion_config)
            
            # map diffusion hyperparameters to gpu
            for key in self.diffusion_hyperparams:
                if key != "T":
                    self.diffusion_hyperparams[key] = self.diffusion_hyperparams[key].cuda()


            # load checkpoint if any
            if self.ckpt_epoch == 'max':
                self.ckpt_epoch = find_max_epoch(output_directory)
            if self.ckpt_epoch >= 0:
                try:
                    # load checkpoint file
                    model_path = os.path.join(output_directory, '{}.pkl'.format(self.ckpt_epoch))
                    checkpoint = torch.load(model_path, map_location='cpu')

                    # feed model dict and optimizer state
                    self.net.load_state_dict(checkpoint['model_state_dict'])
                    if 'optimizer_state_dict' in checkpoint:
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                    print('Successfully loaded model at epoch {}.'.format(self.ckpt_epoch))

                except:
                    self.ckpt_epoch = -1
                    print('No valid checkpoint model found.')
            else:
                self.ckpt_epoch = -1
                print('No valid checkpoint model found.')


    def train(self, data, obs_mask):

        # train eval split
        train, eval, train_obs_mask, eval_obs_mask = train_test_split(
            data, obs_mask, test_size=self.train_config["val_size"]
        )

        # train data loading
        train_ds = TS_Dataset(
            train,
            train_obs_mask,
            "Training",
            self.train_config["missing_mode"],
            self.train_config["missing_r"]
        )
        train_loader = DataLoader(
            train_ds, shuffle=True,
            batch_size=self.train_config["batch_size"]
        )
        
        # eval data loading
        if self.train_config["val_size"]:
            eval_ds = train_ds = TS_Dataset(
                eval,
                eval_obs_mask,
                "Eval",
                self.train_config["missing_mode"],
                self.train_config["missing_r"]
            )
            eval_loader = DataLoader(
                eval_ds, shuffle=True,
                batch_size=self.train_config["batch_size"]
            )
        else:
            eval_loader = None
        print('Data loaded')
        





        # training
        for epoch in tqdm(range(self.ckpt_epoch + 1, self.train_config["epochs"] + 1),
                          total = self.train_config["epochs"] + 1,
                          initial = self.ckpt_epoch + 1,
                          desc="Training the network"):

            # train step
            train_loss = self._step(train_loader, epoch, "Training")

            # evalutation step
            val_loss = "NA"
            if eval_loader:
                val_loss = self._step(eval_loader, epoch, "Validation")

            # output
            if self.train_config["verbose"]:
                    
                string =  "Epoch: {}/{}.. Training Loss: {:.2E}.. Validation Loss: {:.2E}.."
                string = string.format(epoch+1, self.train_config["epochs"],
                                       Decimal(str(train_loss)), Decimal(str(val_loss)))
                tqdm.write(string)

            # save checkpoint
            if epoch > 0 and epoch % self.train_config["epochs_per_ckpt"] == 0:

                checkpoint_name = '{}.pkl'.format(epoch)
                torch.save(
                    {'model_state_dict': self.net.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()},
                    os.path.join(self.train_config["output_directory"], checkpoint_name)
                )
                tqdm.write('Model at epoch %s is saved' % epoch)


    def _step(self, data_loader, epoch, phase):

        if phase == "Training":
            self.net.train()
        else:
            self.net.eval()

        avg_loss = 0.0

        for batch, obs_mask, cond_mask in tqdm(data_loader, total=len(data_loader), leave=False,
                                 desc="Epoch " + str(epoch) + ", " + phase + " phase"):
            
            loss_mask = torch.logical_xor(cond_mask, obs_mask)
            cond_mask = cond_mask.cuda()
            loss_mask = loss_mask.cuda()

            assert batch.size() == cond_mask.size() == loss_mask.size()

            # training
            if phase == "Training":

                self.optimizer.zero_grad()
                X = batch, batch * cond_mask, cond_mask, loss_mask
                loss = training_loss(self.net, self.loss, X, self.diffusion_hyperparams,
                                    only_generate_missing=self.train_config["only_generate_missing"])

                loss.backward()
                self.optimizer.step()

            # evaluating
            else:
                with torch.no_grad():
                    imputed = sampling(
                        self.net,
                        self.diffusion_hyperparams,
                        cond = batch * obs_mask,
                        mask = loss_mask.float(),
                        only_generate_missing=1
                    )
                    
                    loss = self.loss(imputed[loss_mask], batch[loss_mask])

            avg_loss += loss.item()


        avg_loss /= len(data_loader)
        avg_loss=float(avg_loss)

        return avg_loss
    
    def predict(self, data, obs_mask, batch_size):
        
        # create data loader
        dataset = TS_Dataset(
            data,
            obs_mask,
            "Inference"
        )
        data_loader = DataLoader(
            dataset, shuffle=False,
            batch_size = batch_size
        )

        results = []

        # sampling
        for batch, obs_mask in tqdm(
            data_loader, total=len(data_loader),
            leave=False, desc="Sampling"
        ):
            imputed = sampling(
                        self.net,
                        self.diffusion_hyperparams,
                        cond = batch * obs_mask,
                        mask = obs_mask.float(),
                        only_generate_missing=1
                    )
            results.append[imputed]
        
        return np.concatenate(results)
                    