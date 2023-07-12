#-*- coding:utf-8 -*-

from diffusion import create_edm_model, rand_log_normal, Denoiser, ModelType
from diffusion.denoisers import Denoiser, get_sigmas_karras, sample_dpmpp_2m, sample_heun
from diffusers.models import AutoencoderKL
from torchvision.utils import make_grid
from torchvision import transforms as T
from dataset import FolderDataset, expand2square

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
parser.add_argument('--in_channels', type=int, default=4)
parser.add_argument('--out_channels', type=int, default=4)

# Karras (EDM) options
parser.add_argument('--sigma_data', type=float, default=0.5)
parser.add_argument('--sigma_sample_density_mean', type=float, default=-1.2)
parser.add_argument('--sigma_sample_density_std', type=float, default=1.2)
parser.add_argument('--sigma_max', type=float, default=80)
parser.add_argument('--sigma_min', type=float, default=0.0002)
parser.add_argument('--rho', type=float, default=7.0)

# Resuming options
parser.add_argument('--resume', action='store_true')
parser.add_argument('--resume_checkpoint', type=str)
parser.add_argument('--resume_epochs', type=int, default=0)
opt = parser.parse_args()

device = 'cuda'
LATENT_DIM = opt.image_size // 8

def create_inner_model(model_type:ModelType = ModelType.CNN):
    # create network object
    if model_type == ModelType.CNN:
        inner_model = create_edm_model(
            image_size=LATENT_DIM, # For VAE latent
            num_channels=opt.num_channels,
            num_res_blocks=opt.num_res_blocks,
            in_channels=opt.in_channels,
            out_channels=opt.out_channels
        )
    elif model_type == ModelType.TRANSFORMER:
        raise NotImplementedError
    return inner_model

def train():
    with tqdm(range(opt.resume_epochs, opt.epochs), desc='Epoch') as tglobal:
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
                    nlatent = encode_img(nimage)
                    B = nlatent.shape[0]

                    # sample noise to add to actions
                    noise = torch.randn(nlatent.shape, device=device)

                    # sample a diffusion iteration for each data point
                    sigmas = sample_density([B], device=device)
                        
                    # # L2 loss
                    loss = noise_pred_net.loss(nlatent, noise, sigmas, global_cond=None)
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
                torch.save(noise_pred_net.state_dict(), f'{opt.export_folder}/[training]edm-latent-diffusion-{opt.model_type}-epoch{epoch_idx+1}'+'.pt')
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

        torch.save(noise_pred_net.state_dict(), f'{opt.export_folder}/edm-latent-diffusion-{opt.model_type}-epoch{opt.epochs+1}'+'.pt')

@torch.no_grad()
def generate(ema_noise_pred_net, model_type:ModelType, sigma_max:float, sigma_min:float, rho:float, num_diffusion_iters:int, export_name:str, sample_num:int, device:str='cuda'):
    B = sample_num
    obs_cond = None
    with torch.no_grad():
        # initialize action from Guassian noise
        noisy_latent = torch.randn(
            (B, opt.in_channels, LATENT_DIM, LATENT_DIM), device=device)
        nlatent = noisy_latent * sigma_max
        sigmas = get_sigmas_karras(num_diffusion_iters, sigma_min, sigma_max, rho=rho, device=device)
        nlatent = sample_dpmpp_2m(ema_noise_pred_net, nlatent, sigmas, disable=True, extra_args={'global_cond':obs_cond})
        nimage = decode_img(nlatent)
        imgs = nimage.detach().to('cpu')
        
        img = make_grid(imgs)
        img = transforms.functional.to_pil_image(img)
        # (B, 3, H, W)
    img = Image.fromarray(np.asarray(img))
    img.save(export_name)

def encode_img(img:torch.Tensor):
    z = vae.encode(img).latent_dist.sample().detach() # z : (B, 4, 32, 32)
    return z 

def decode_img(z:torch.Tensor):
    x = vae.decode(z).sample 
    return x 

if __name__ == '__main__':
    image_transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Lambda(lambda img: expand2square(img)),
        T.Resize(opt.image_size),
        T.CenterCrop(opt.image_size),
        T.ToTensor(),
    ])
    dataset = FolderDataset(
        folder_path=opt.dataset_path,
        image_size=opt.image_size,
        transform=image_transform
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
    if opt.resume:
        state_dict = torch.load(opt.resume_checkpoint, map_location='cuda')
        noise_pred_net.load_state_dict(state_dict)
        print("Pretrained Model Loaded")

    sample_density = partial(rand_log_normal, loc=opt.sigma_sample_density_mean, scale=opt.sigma_sample_density_std)

    ema = EMAModel(
        model=noise_pred_net,
        power=opt.ema_power,
        parameters=noise_pred_net.parameters(),
    )

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    print("VAE Loaded!")

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

    # TODO: fix wandb resume
    run = wandb.init(project = 'edm_latent_diffusion', resume = opt.resume)
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