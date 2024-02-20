import random
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

import submodules.gaussian_editor.threestudio as threestudio
from submodules.gaussian_editor.threestudio.utils.base import BaseModule
from submodules.gaussian_editor.threestudio.utils.typing import *


class BaseBackground(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pass

    cfg: Config

    def configure(self):
        pass

    def forward(self, dirs: Float[Tensor, "B H W 3"]) -> Float[Tensor, "B H W Nc"]:
        raise NotImplementedError
