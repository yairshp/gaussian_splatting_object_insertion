import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

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
 

def main():
    # with torch.cuda.device(1):
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

    scene_views_dataset = SceneViewsDataset(params["dataset_params"], params["iteration"])
    dataloader = DataLoader(scene_views_dataset, batch_size=None, shuffle=False)

    object_inserter = ObjectInserter(fg_gaussians, bg_gaussians, params["pipeline_params"], params["training_params"])
    object_inserter.to('cuda')

    optimizer = Adam(object_inserter.parameters(), lr=LR)
    
    for view in dataloader:
        optimizer.zero_grad()
        rendering = object_inserter(view)
        # loss = losses.vae_reconstrucion_loss(rendering)
        loss = losses.diffusion_reconstruction_loss(rendering.to('cuda'))
        # loss = losses.diffusion_reconstruction_loss(rendering.to('cpu'))
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    main()