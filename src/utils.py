from argparse import ArgumentParser
from omegaconf import OmegaConf
from submodules.gaussian_editor.gaussiansplatting.arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from submodules.gaussian_editor.gaussiansplatting.utils.general_utils import safe_state

def get_gaussians_params(gaussians_path):
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