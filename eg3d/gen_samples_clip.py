# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Generate images and shapes using pretrained network pickle."""

from itertools import zip_longest
import os
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm
import mrcfile

from criteria.clip_loss import CLIPLoss
from criteria.id_loss import IDLoss
import clip
from torch import optim
import math


import legacy
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from torch_utils import misc
from training.triplane import TriPlaneGenerator,TriPlaneGenerator2
from torch.utils.tensorboard import SummaryWriter




#----------------------------------------------------------------------------
global G
G=None

def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        if m := range_re.match(p):
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')

#----------------------------------------------------------------------------

def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

#----------------------------------------------------------------------------

def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--shapes', help='Export shapes as .mrc files viewable in ChimeraX', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--shape-res', help='', type=int, required=False, metavar='int', default=512, show_default=True)
@click.option('--fov-deg', help='Field of View of camera in degrees', type=int, required=False, metavar='float', default=18.837, show_default=True)
@click.option('--shape-format', help='Shape Format', type=click.Choice(['.mrc', '.ply']), default='.mrc')
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)

#for clip loss, default="a person with black hair"
@click.option('--step', type=int, default=300, help="number of optimization steps", show_default=True)
@click.option('--save_freq', type=int, default=300, show_default=True)
@click.option("--neutral", type=str, default="a person with hair", help="the text that guides the editing/generation")
@click.option("--target", type=str, default="a person with red hair", help="the text that guides the editing/generation")
@click.option("--init_lr", type=float, default=0.1, help="learning rate") 
@click.option("--l2_lambda", type=float, default=0.008, help="weight of the latent distance (used for editing only)") 
@click.option("--id_lambda", type=float, default=0.005, help="weight of id loss (used for editing only)")
@click.option('--z_edit', help='edit from z', type=bool, required=False, default=False, show_default=True)
@click.option('--view', help='render result from other view', type=bool, required=False, default=False, show_default=True)
@click.option('--tensor', help='tensorboard', type=bool, required=False, default=False, show_default=True)
@click.option('--mod', help='tensorboard', type=bool, required=False, default=False, show_default=True)


def generate_images(
    network_pkl: str,
    seeds: List[int],
    truncation_psi: float,
    truncation_cutoff: int,
    outdir: str,
    shapes: bool,
    shape_res: int,
    fov_deg: float,
    shape_format: str,
    class_idx: Optional[int],
    reload_modules: bool,
    step: int,
    save_freq: int,
    neutral: str,
    target: str,
    init_lr: float,
    l2_lambda: float,
    id_lambda: float,
    z_edit: bool,
    view: bool,
    tensor: bool,
    mod: bool
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate an image using pre-trained FFHQ model.
    python gen_samples.py --outdir=output --trunc=0.7 --seeds=0-5 --shapes=True\\
        --network=ffhq-rebalanced-128.pkl
    """
    if tensor:
        writer = SummaryWriter()
    """
    import pickle
    with open('./networks/ffhq512-128.pkl', 'rb') as f: 
        p = pickle.Unpickler(f) 
        data = p.load() # 여기서 에러나면 데이터가 유효하지 않은 것
        f.close()
    """
    global G

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    mod_tri = TriPlaneGenerator2()
    setattr(G, 'sym', mod_tri.sym)
    del mod_tri

    # Specify reload_modules=True if you want code modifications to take effect; otherwise uses pickled code
    if reload_modules:
        print("Reloading Modules!")
        G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G_new.neural_rendering_resolution = G.neural_rendering_resolution
        G_new.rendering_kwargs = G.rendering_kwargs
        G = G_new

    os.makedirs(outdir, exist_ok=True)    

    cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, torch.tensor([0, 0, 0.2], device=device), radius=2.7, device=device)
    intrinsics = FOV_to_intrinsics(fov_deg, device=device)

    #clip description 받기
    neutral_inputs = torch.cat([clip.tokenize(neutral)]).cuda()
    target_inputs = torch.cat([clip.tokenize(target)]).cuda()
    classnames=[target_inputs, neutral_inputs]

    model_choice = ["ViT-B/32", "ViT-B/16"]
    model_weights = [1.0, 0.0]

    # clip loss
    clip_loss_models = {model_name: CLIPLoss(clip_model=model_name) for model_name in model_choice}
    clip_model_weights = {model_name: weight for model_name, weight in zip(model_choice, model_weights)}

    id_loss = IDLoss()

    # Generate images.
    for seed_idx, seed in enumerate(seeds): # seeds = [0,1,2,3]
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        orig_z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device) # z가 random seed(latent code) z shape는 (1,512)

        if z_edit:
            z = orig_z.detach().clone()
            z.requires_grad = True

            optimizer = optim.Adam([z], init_lr)

        cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
        cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
        conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
        conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        orig_ws = G.mapping(orig_z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff) # ws shape [1,14,512]

        if not z_edit:
            ws = orig_ws.detach().clone()
            ws.requires_grad = True
            optimizer = optim.Adam([ws], init_lr)

        orig_imgs = []
        angle_p = -0.2
        pbar = tqdm(range(step))
        for i in pbar:
            fin_loss=0.0
            t = i / step
            imgs = []
            imgs_mod = []
            for angle_y, angle_p in [(.4, angle_p), (0, angle_p), (-.4, angle_p)]:
                cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)
                camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
                
                orig_img = G.synthesis(orig_ws, camera_params)['image'] 
                
                if i==0:
                    orig_img_to = (orig_img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                    orig_imgs.append(orig_img_to)
                    # styleclip에서 clip으로 들어갈 때 img shape = [25,3,256,256]

                lr = get_lr(t, init_lr)
                optimizer.param_groups[0]["lr"] = lr

                if z_edit:
                    ws = G.mapping(z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff) # ws shape [1,14,512]
                    img = G.synthesis(ws, camera_params)['image']
                else:
                    if not mod:
                        img = G.synthesis(ws, camera_params)['image']
                    else:
                        img_mod = G.sym(G, ws, orig_ws, camera_params)['image']
                        img = G.synthesis(ws, camera_params)['image']

                # clip loss, identity loss, l2 loss 모두 더하면 final loss
                c_loss = torch.sum(torch.stack([clip_model_weights[model_name] * clip_loss_models[model_name](orig_img, neutral, img, target) 
                                                    for model_name in clip_model_weights.keys()]))
                i_loss = id_loss(img, orig_img)[0]
                if z_edit:
                    l2_loss = ((orig_z - z) ** 2).sum()
                else:
                    l2_loss = ((orig_ws - ws) ** 2).sum()
                loss = c_loss + l2_lambda * l2_loss + id_lambda * i_loss
                fin_loss+=loss

                if not mod:
                    if i!=0 and i%save_freq==0:
                        img_to = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                        imgs.append(img_to)
                        img = torch.cat(imgs, dim=2)
                        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/edit{seed:04d}_{i:04d}.png')
                else:
                    if i!=0 and i%save_freq==0:
                        img_to = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                        imgs.append(img_to)
                        img = torch.cat(imgs, dim=2)
                        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/edit{seed:04d}_{i:04d}.png')

                        img_to_mod = (img_mod.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                        imgs_mod.append(img_to_mod)
                        img_mod = torch.cat(imgs_mod, dim=2)
                        PIL.Image.fromarray(img_mod[0].cpu().numpy(), 'RGB').save(f'{outdir}/edit_mod{seed:04d}_{i:04d}.png')

            optimizer.zero_grad()
            fin_loss.backward()
            optimizer.step()
            if tensor:
                writer.add_scalar(f"Loss/{seed:04d}", fin_loss, i)

            pbar.set_description(
                (
                    f"loss: {fin_loss.item():.4f};"
                )
            )                
        orig_img = torch.cat(orig_imgs, dim=2)
        PIL.Image.fromarray(orig_img[0].cpu().numpy(), 'RGB').save(f'{outdir}/orig{seed:04d}.png')


        if shapes:
            # extract a shape.mrc with marching cubes. You can view the .mrc file using ChimeraX from UCSF.
            max_batch=1000000

            samples, voxel_origin, voxel_size = create_samples(N=shape_res, voxel_origin=[0, 0, 0], cube_length=G.rendering_kwargs['box_warp'] * 1)#.reshape(1, -1, 3)
            
            samples = samples.to(orig_z.device)
            sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=orig_z.device)
            transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=orig_z.device)
            transformed_ray_directions_expanded[..., -1] = -1

            head = 0
            with tqdm(total = samples.shape[1]) as pbar:
                with torch.no_grad():
                    while head < samples.shape[1]:
                        torch.manual_seed(0)
                        sigma = G.sample(samples[:, head:head+max_batch], transformed_ray_directions_expanded[:, :samples.shape[1]-head], orig_z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, noise_mode='const')['sigma']
                        sigmas[:, head:head+max_batch] = sigma
                        head += max_batch
                        pbar.update(max_batch)

            sigmas = sigmas.reshape((shape_res, shape_res, shape_res)).cpu().numpy()
            sigmas = np.flip(sigmas, 0)

            # Trim the border of the extracted cube
            pad = int(30 * shape_res / 256)
            pad_value = -1000
            sigmas[:pad] = pad_value
            sigmas[-pad:] = pad_value
            sigmas[:, :pad] = pad_value
            sigmas[:, -pad:] = pad_value
            sigmas[:, :, :pad] = pad_value
            sigmas[:, :, -pad:] = pad_value

            if shape_format == '.ply':
                from shape_utils import convert_sdf_samples_to_ply
                convert_sdf_samples_to_ply(np.transpose(sigmas, (2, 1, 0)), [0, 0, 0], 1, os.path.join(outdir, f'orig{seed:04d}.ply'), level=10)
            elif shape_format == '.mrc': # output mrc
                with mrcfile.new_mmap(os.path.join(outdir, f'orig{seed:04d}.mrc'), overwrite=True, shape=sigmas.shape, mrc_mode=2) as mrc:
                    mrc.data[:] = sigmas
            

            # z_edit을 하면 z가 edit 되기 때문에 하나 더 생성 가능
            max_batch=1000000

            samples, voxel_origin, voxel_size = create_samples(N=shape_res, voxel_origin=[0, 0, 0], cube_length=G.rendering_kwargs['box_warp'] * 1)#.reshape(1, -1, 3)
            
            samples = samples.to(orig_z.device)
            sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=orig_z.device)
            transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=orig_z.device)
            transformed_ray_directions_expanded[..., -1] = -1

            head = 0
            with tqdm(total = samples.shape[1]) as pbar:
                with torch.no_grad():
                    while head < samples.shape[1]:
                        torch.manual_seed(0)
                        if z_edit:
                            sigma = G.sample(samples[:, head:head+max_batch], transformed_ray_directions_expanded[:, :samples.shape[1]-head], z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, noise_mode='const')['sigma']
                        else:
                            sigma = G.sample_mixed(samples[:, head:head+max_batch], transformed_ray_directions_expanded[:, :samples.shape[1]-head], ws, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, noise_mode='const')['sigma']
                        sigmas[:, head:head+max_batch] = sigma
                        head += max_batch
                        pbar.update(max_batch)

            sigmas = sigmas.reshape((shape_res, shape_res, shape_res)).cpu().numpy()
            sigmas = np.flip(sigmas, 0)

            # Trim the border of the extracted cube
            pad = int(30 * shape_res / 256)
            pad_value = -1000
            sigmas[:pad] = pad_value
            sigmas[-pad:] = pad_value
            sigmas[:, :pad] = pad_value
            sigmas[:, -pad:] = pad_value
            sigmas[:, :, :pad] = pad_value
            sigmas[:, :, -pad:] = pad_value

            if shape_format == '.ply':
                from shape_utils import convert_sdf_samples_to_ply
                convert_sdf_samples_to_ply(np.transpose(sigmas, (2, 1, 0)), [0, 0, 0], 1, os.path.join(outdir, f'edit{seed:04d}.ply'), level=10)
            elif shape_format == '.mrc': # output mrc
                with mrcfile.new_mmap(os.path.join(outdir, f'edit{seed:04d}.mrc'), overwrite=True, shape=sigmas.shape, mrc_mode=2) as mrc:
                    mrc.data[:] = sigmas
    
    if tensor:
        writer.close()


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
