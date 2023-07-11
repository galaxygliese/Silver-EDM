#-*- coding:utf-8 -*-

from diffusion import create_edm_model, rand_log_normal, Denoiser, ModelType
from diffusion.denoisers import Denoiser, get_sigmas_karras, sample_dpmpp_2m, sample_heun
from torchvision.utils import make_grid
from torchvision.transforms import Compose, Lambda
from dataset import FolderDataset

from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler

import torchvision.transforms as transforms
from functools import partial
from copy import deepcopy
from tqdm.auto import tqdm
from PIL import Image
import numpy as np
import argparse
import torch.nn as nn
import torch
import wandb
import os 


parser = argparse.ArgumentParser()

# General options
parser.add_argument('-e', '--epochs', type=int, default=100)
parser.add_argument('-b', '--batchsize', type=int, default=8)
parser.add_argument('--diffusion_timesteps', type=int, default=40) # Different from training
parser.add_argument('--ema_power', type=float, default=0.75)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('-m', '--model_type', type=str, default="C")
parser.add_argument('--export_folder', type=str, default="./checkpoints")
parser.add_argument('--warmup_steps', type=int, default=50)
parser.add_argument('--num_channels', type=int, default=64)
parser.add_argument('--num_res_blocks', type=int, default=1)
parser.add_argument('--save-per-epoch', type=int, default=5)

# Dataset options
parser.add_argument('--dataset_path', type=str)
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--in_channels', type=int, default=3)
parser.add_argument('--out_channels', type=int, default=3)

# Karras (EDM) options
parser.add_argument('--sigma_data', type=float, default=0.5)
parser.add_argument('--sigma_sample_density_mean', type=float, default=-1.2)
parser.add_argument('--sigma_sample_density_std', type=float, default=1.2)
parser.add_argument('--sigma_max', type=float, default=80)
parser.add_argument('--sigma_min', type=float, default=0.0002)
parser.add_argument('--rho', type=float, default=7.0)
opt = parser.parse_args()

device = 'cuda'

def create_inner_model(model_type:ModelType = ModelType.CNN):
    # create network object
    if model_type == ModelType.CNN:
        inner_model = create_edm_model(
            image_size=opt.image_size,
            num_channels=opt.num_channels,
            num_res_blocks=opt.num_res_blocks,
            in_channels=opt.in_channels,
            out_channels=opt.out_channels
        )
    elif model_type == ModelType.TRANSFORMER:
        raise NotImplementedError
    return inner_model

def train():
    with tqdm(range(opt.epochs), desc='Epoch') as tglobal:
        # epoch loop
        for epoch_idx in tglobal:
            epoch_loss = list()
            noise_pred_net.train()
            # batch loop
            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    # data normalized in dataset
                    # device transfer
                    nimage = nbatch.to(device)
                    B = nimage.shape[0]

                    # sample noise to add to actions
                    noise = torch.randn(nimage.shape, device=device)

                    # sample a diffusion iteration for each data point
                    sigmas = sample_density([B], device=device)
                        
                    # # L2 loss
                    loss = noise_pred_net.loss(nimage, noise, sigmas, global_cond=None)
                    loss = loss.mean()

                    # optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    # step lr scheduler every batch
                    # this is different from standard pytorch behavior
                    lr_scheduler.step()

                    # update Exponential Moving Average of the model weights
                    ema.step(inner_model)

                    # logging
                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)

            run.log({"train-loss": np.mean(epoch_loss)})
            tglobal.set_postfix(loss=np.mean(epoch_loss))   
        
            if (epoch_idx + 1) % opt.save_per_epoch == 0:
                torch.save(noise_pred_net.state_dict(), f'{opt.export_folder}/[training]edm-diffusion-{opt.model_type}-epoch{epoch_idx+1}'+'.pt')
                noise_pred_net.eval()
                generate(
                    ema_noise_pred_net=noise_pred_net,
                    model_type=model_type,
                    sigma_max=opt.sigma_max,
                    sigma_min=opt.sigma_min,
                    rho=opt.rho,
                    num_diffusion_iters=num_diffusion_iters,
                    export_name=f"{opt.export_folder}/epoch{epoch_idx+1}.png",
                    sample_num=4
                )

        torch.save(noise_pred_net.state_dict(), f'{opt.export_folder}/t-push-edm-diffusion-{opt.model_type}-epoch{epoch_idx+1}'+'.pt')

@torch.no_grad()
def generate(ema_noise_pred_net, model_type:ModelType, sigma_max:float, sigma_min:float, rho:float, num_diffusion_iters:int, export_name:str, sample_num:int, device:str='cuda'):
    B = sample_num
    obs_cond = None
    with torch.no_grad():
        # initialize action from Guassian noise
        noisy_image = torch.randn(
            (B, opt.in_channels, opt.image_size, opt.image_size), device=device)
        nimage = noisy_image * sigma_max
        sigmas = get_sigmas_karras(num_diffusion_iters, sigma_min, sigma_max, rho=rho, device=device)
        nimage = sample_dpmpp_2m(ema_noise_pred_net, nimage, sigmas, disable=True, extra_args={'global_cond':obs_cond})
        nimage = nimage.detach().to('cpu')
        
        imgs = 0.5*(nimage+1)
        img = make_grid(imgs)
        img = transforms.functional.to_pil_image(img)
        # (B, 3, H, W)
    img = Image.fromarray(np.asarray(img))
    img.save(export_name)
    print("Done!")

if __name__ == '__main__':
    dataset = FolderDataset(
        folder_path=opt.dataset_path,
        image_size=opt.image_size,
    )

    # create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batchsize,
        num_workers=1,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True, 
        # don't kill worker process afte each epoch
        persistent_workers=True 
    )

    # visualize data in batch
    batch = next(iter(dataloader))
    print("batch.shape:", batch.shape)
    num_diffusion_iters = opt.diffusion_timesteps

    # device transfer
    model_type = ModelType.CNN if opt.model_type == 'C' else ModelType.TRANSFORMER
    print("Model Type:", model_type)
    inner_model = create_inner_model(model_type=model_type)
    device = torch.device('cuda')
    _ = inner_model.to(device)
    noise_pred_net = Denoiser(inner_model=inner_model, sigma_data=opt.sigma_data)

    sample_density = partial(rand_log_normal, loc=opt.sigma_sample_density_mean, scale=opt.sigma_sample_density_std)

    ema = EMAModel(
        model=noise_pred_net,
        power=opt.ema_power,
        parameters=noise_pred_net.parameters(),
    )

    optimizer = torch.optim.AdamW(
        params=inner_model.parameters(), 
        lr=opt.lr, weight_decay=1e-6
    )

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=opt.warmup_steps,
        num_training_steps=len(dataloader) * opt.epochs
    )

    run = wandb.init(project = 'edm_diffusion')
    config = run.config
    config.epochs = opt.epochs
    config.batchsize = opt.batchsize
    config.learning_rate = opt.lr 
    config.diffusion_timesteps = opt.diffusion_timesteps
    config.model_type = opt.model_type

    config.sigma_data = opt.sigma_data
    config.sigma_sample_density_mean = opt.sigma_sample_density_mean
    config.sigma_sample_density_std = opt.sigma_sample_density_std

    train()