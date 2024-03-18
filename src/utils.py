import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from argparse import ArgumentParser
from omegaconf import OmegaConf
from submodules.gaussian_editor.gaussiansplatting.arguments import (
    ModelParams,
    PipelineParams,
    OptimizationParams,
    get_combined_args,
)
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
        parser=ArgumentParser(description="Training script parameters"),
        max_steps=1500,
        lr_scaler=3.0,
        lr_final_scaler=2.0,
        color_lr_scaler=3.0,
        opacity_lr_scaler=2.0,
        scaling_lr_scaler=2.0,
        rotation_lr_scaler=2.0,
    )
    training_params = OmegaConf.create(vars(training_params))
    return {
        "pipeline_params": pipeline_params,
        "dataset_params": dataset_params,
        "training_params": training_params,
        "iteration": args.iteration,
    }


# rotation_matrix_chair = torch.from_numpy(
#     R.from_rotvec(-np.pi / 2 * np.array([1.1, 0.0, 0.0])).as_matrix()
# )

rotation_matrix_chair = torch.tensor(
    np.array(
        [
            [-0.06443945318460464, 0.26465365290641785, -0.9621880650520326],
            [0.9979217052459716, 0.017089653760194775, -0.06213200092315674],
            [3.338260770488323e-18, -0.9641920328140258, -0.2652048468589782],
        ]
    )
)

rotation_matrix_trash_can = torch.tensor(
    np.array(
        [
            [0.0005396426422521473, 0.0012905767653137443, -0.004969370551407337],
            [0.005134222097694874, -0.00013564866094384342, 0.0005223156185820699],
            [-9.615913113045222e-12, -0.004996745381504297, -0.0012976857833564281],
        ]
    )
)
