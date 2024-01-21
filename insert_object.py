import wandb
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms
from pytorch3d.transforms import axis_angle_to_matrix
from kornia.geometry.quaternion import Quaternion

from omegaconf import OmegaConf
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from utils.general_utils import safe_state

from scene.gaussian_model_concatable import GaussianModelConcatable
from scene.vanilla_gaussian_model import GaussianModel as VanillaGaussianModel

import losses
from models.object_inserter import ObjectInserter
from data.scene_views_dataset import SceneViewsDataset

BG_SCENE_PATH = '/root/projects/insert_object/data/garden/output'
BG_GAUSSIANS_PATH = '/root/projects/insert_object/data/garden/output/point_cloud/iteration_7000/point_cloud.ply'
FG_GAUSSIANS_PATH = '/root/projects/insert_object/data/hotdog/output/point_cloud/iteration_30000/point_cloud.ply'
LR = 0.0001
EPOCHS = 20

@torch.no_grad()
def scale_gaussians(gaussian, scale):
    gaussian._xyz.data = gaussian._xyz.data * scale
    g_scale = gaussian.get_scaling * scale
    gaussian._scaling.data = torch.log(g_scale + 1e-7)


@torch.no_grad()
def rotate_gaussians(gaussian, rotmat):
    rot_q = Quaternion.from_matrix(rotmat[None, ...])
    g_qvec = Quaternion(gaussian.get_rotation)
    gaussian._rotation.data = (rot_q * g_qvec).data

    gaussian._xyz.data = torch.einsum("ij,bj->bi", rotmat, gaussian._xyz.data)


@torch.no_grad()
def translate_gaussians(gaussian, tvec):
    gaussian._xyz.data = gaussian._xyz.data + tvec[None, ...]
def get_params(gaussians_path):
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline_params = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser, f"-m {gaussians_path}".split())
    safe_state(args.quiet)
    pipeline_params = pipeline_params.extract(args)
    dataset_params = model.extract(args)
    training_params = OptimizationParams(
            parser = ArgumentParser(description="Training script parameters"),
            max_steps= 1500,
            lr_scaler = 3.0,
            lr_final_scaler = 2.0,
            color_lr_scaler = 3.0,
            opacity_lr_scaler = 2.0,
            scaling_lr_scaler = 2.0,
            rotation_lr_scaler = 2.0,
        )
    training_params = OmegaConf.create(vars(training_params))
    return {
        "pipeline_params": pipeline_params,
        "dataset_params": dataset_params,
        "training_params": training_params,
        "iteration": args.iteration
    }
 
def get_cuda_tensors():
    tensors = []
    for obj in gc.get_objects():
        try:
            if (torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data))) and 'cpu':
                tensors.append(obj)
        except:
            pass
    return tensors

def disable_grads(gaussians):
    for attr_name in dir(gaussians):
        attr = getattr(gaussians, attr_name)
        if torch.is_tensor(attr) and attr.requires_grad:
            attr.requires_grad = False


def main():
    # with torch.cuda.device(0):
    wandb.init(project="insert-object")
    params = get_params(BG_SCENE_PATH)
    fg_gaussians = GaussianModelConcatable(
                sh_degree=0,
                anchor_weight_init_g0=1.0,
                anchor_weight_init=0.1,
                anchor_weight_multiplier=2,
            )
    fg_gaussians.load_ply(FG_GAUSSIANS_PATH)
    bg_gaussians = VanillaGaussianModel(fg_gaussians.max_sh_degree)
    bg_gaussians.load_ply(BG_GAUSSIANS_PATH)

    scale_gaussians(fg_gaussians, 0.4)

    tvec = torch.tensor([1.2, 3.5, -1.2], requires_grad=True).to('cuda')
    translate_gaussians(fg_gaussians, tvec)

    rotation_axes = torch.tensor([torch.pi/1.5, 0, 0]).to('cuda')
    rotation_matrix = axis_angle_to_matrix(rotation_axes)
    rotate_gaussians(fg_gaussians, rotation_matrix)

    disable_grads(fg_gaussians)

    scene_views_dataset = SceneViewsDataset(params["dataset_params"], params["iteration"])
    dataloader = DataLoader(scene_views_dataset, batch_size=None, shuffle=False)

    object_inserter = ObjectInserter(fg_gaussians, bg_gaussians, params["pipeline_params"], params["training_params"])
    object_inserter.to('cuda')

    optimizer = Adam(object_inserter.parameters(), lr=LR)
    
    to_img = transforms.ToPILImage()

    for epoch in range(EPOCHS):
        # avg_loss = torch.tensor(0, dtype=torch.float32)
        imgs = []
        for i, view in enumerate(dataloader):
            optimizer.zero_grad()
            rendering = object_inserter(view).to('cpu')
            # loss = losses.vae_reconstrucion_loss(rendering)
            # loss = losses.diffusion_reconstruction_loss(rendering.to('cuda:1'))
            loss = losses.diffusion_reconstruction_loss(rendering.to('cpu'))
            # loss = losses.debug_loss(rendering)
            # avg_loss += loss / torch.tensor(len(dataloader), dtype=loss.dtype, device=loss.device)
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
        # avg_loss.backward()
        # optimizer.step()
            print(f"Epoch: {epoch + 1}, View: {i + 1}, Loss: {loss}")
            wandb.log({'loss': loss})
            if epoch % 3 == 0:
                imgs.append(wandb.Image(to_img(rendering)))
                if i == len(dataloader) - 1:
                    wandb.log({'renderings': imgs})
        

if __name__ == "__main__":
    main()