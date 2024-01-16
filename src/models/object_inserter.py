import torch
import torch.nn as nn

from gaussian_renderer import render

# typing related imports
from scene.gaussian_model_concatable import GaussianModelConcatable
from scene.vanilla_gaussian_model import VanillaGaussianModel
from arguments import PipelineParams

class ObjectInserter(nn.Module):
    def __init__(self, original_fg_gaussian: GaussianModelConcatable, bg_gaussian: VanillaGaussianModel, pipeline_params: PipelineParams, training_params: dict):
        super(ObjectInserter, self).__init__()

        self.training_params = training_params
        self.pipeline_params = pipeline_params
        self.background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

        self.original_fg_gaussian = original_fg_gaussian
        self.original_fg_gaussian.training_setup(self.training_params)
        self.bg_gaussian = bg_gaussian
        self.curr_fg_gaussian = None
        self.reset_fg_gaussian()

        self.tvec = nn.Parameter(torch.zeros(3), requires_grad=True)

    def forward(self, view):
        self.curr_fg_gaussian.concat_gaussians(self.bg_gaussian, self.tvec)
        rendering = render(view, self.curr_fg_gaussian, self.pipeline_params, self.background)['render']
        self.reset_fg_gaussian()  # resetting because the current is transformed and concatenated
        return rendering

    def reset_fg_gaussian(self):
        self.curr_fg_gaussian = self.original_fg_gaussian
        self.curr_fg_gaussian.training_setup(self.training_params)