#-*- coding:utf-8 -*-

from torchvision.utils import make_grid, save_image
from diffusion.denoisers import Denoiser, get_sigmas_karras, sample_dpmpp_2m
from diffusers.models import AutoencoderKL
from diffusion import create_edm_model, ModelType
from tqdm.auto import tqdm
from PIL import Image
import torchvision.transforms as transforms
import collections
import numpy as np
import argparse
import torch

parser = argparse.ArgumentParser()

# General options
parser.add_argument('-w', '--checkpoint', type=str, default="./checkpoints/edm-diffusion-epoch50.pt")
parser.add_argument('-e', '--export_file', type=str, default="./results/exported-edm-ldm.png")
parser.add_argument('--max_steps', type=int, default=200)
parser.add_argument('--diffusion_timesteps', type=int, default=40)
parser.add_argument('-m', '--model_type', type=str, default="C")

# Dataset options
parser.add_argument('--dataset_path', type=str)
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--in_channels', type=int, default=4)
parser.add_argument('--out_channels', type=int, default=4)
parser.add_argument('--num_channels', type=int, default=64)
parser.add_argument('--num_res_blocks', type=int, default=1)
parser.add_argument('--sample_num', type=int, default=4)

# Karras (EDM) options
parser.add_argument('--sigma_data', type=float, default=0.5)
parser.add_argument('--sigma_sample_density_mean', type=float, default=-1.2)
parser.add_argument('--sigma_sample_density_std', type=float, default=1.2)
parser.add_argument('--sigma_max', type=float, default=80)
parser.add_argument('--sigma_min', type=float, default=0.0002)
parser.add_argument('--rho', type=float, default=7.0)
opt = parser.parse_args()

def create_inner_model(model_type:ModelType = ModelType.CNN):
    # create network object
    if model_type == ModelType.CNN:
        inner_model = create_edm_model(
            image_size=opt.image_size // 8,
            num_channels=opt.num_channels,
            num_res_blocks=opt.num_res_blocks,
            in_channels=opt.in_channels,
            out_channels=opt.out_channels
        )
    elif model_type == ModelType.TRANSFORMER:
        raise NotImplementedError
    return inner_model

def load_model(model_type:ModelType = ModelType.CNN):
    ckpt_path = opt.checkpoint
    device = torch.device('cuda')

    state_dict = torch.load(ckpt_path, map_location='cuda')
    inner_model = create_inner_model(model_type=model_type).to(device)

    ema_noise_pred_net = Denoiser(inner_model=inner_model, sigma_data=opt.sigma_data)
    ema_noise_pred_net.load_state_dict(state_dict)
    ema_noise_pred_net.eval()
    print('Pretrained weights loaded.')
    return ema_noise_pred_net

def load_vae(device:str='cuda'):
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    print("VAE Loaded!")
    return vae

def decode_img(z:torch.Tensor, vae):
    x = vae.decode(z).sample 
    return x 

@torch.no_grad()
def generate(ema_noise_pred_net, vae, model_type:ModelType, sigma_max:float, sigma_min:float, rho:float, num_diffusion_iters:int, export_name:str, sample_num:int, device:str='cuda'):
    B = sample_num
    obs_cond = None
    with torch.no_grad():
        # initialize action from Guassian noise
        noisy_latent = torch.randn(
            (B, opt.in_channels, opt.image_size // 8, opt.image_size // 8), device=device)
        nlatent = noisy_latent * sigma_max
        sigmas = get_sigmas_karras(num_diffusion_iters, sigma_min, sigma_max, rho=rho, device=device)
        nlatent = sample_dpmpp_2m(ema_noise_pred_net, nlatent, sigmas, disable=True, extra_args={'global_cond':obs_cond})
        
        nimage = decode_img(nlatent, vae)
        imgs = nimage.detach().to('cpu')
        img = make_grid(imgs)
        img = transforms.functional.to_pil_image(img)
        # (B, 3, H, W)
    img = Image.fromarray(np.asarray(img))
    img.save(export_name)
    print("Done!")

def main():
    num_diffusion_iters = opt.diffusion_timesteps
    device = torch.device('cuda')

    model_type = ModelType.CNN if opt.model_type == 'C' else ModelType.TRANSFORMER
    print("Model Type:", model_type)
    ema_noise_pred_net = load_model(model_type=model_type)
    vae = load_vae(device=device)
    sigma_max = opt.sigma_max 
    sigma_min = opt.sigma_min 
    rho = opt.rho

    generate(
        ema_noise_pred_net, 
        vae=vae,
        model_type=model_type, 
        sigma_max=sigma_max, 
        sigma_min=sigma_min, 
        rho=rho, 
        num_diffusion_iters=num_diffusion_iters, 
        export_name=opt.export_file, 
        sample_num=opt.sample_num, 
        device=device,
    )

if __name__ == '__main__':
    main()