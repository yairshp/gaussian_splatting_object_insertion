from torch.utils.data import Dataset

from scene import Scene
from scene.vanilla_gaussian_model import GaussianModel as VanillaGaussianModel
from utils.general_utils import safe_state

from arguments import GroupParams


class SceneViewsDataset(Dataset):
    def __init__(self, dataset_params: GroupParams, iteration: int):
        gaussians = VanillaGaussianModel(dataset_params.sh_degree)
        scene = Scene(dataset_params, gaussians, load_iteration=iteration, shuffle=False)
        self.scene_views = scene.getTrainCameras()

    def __len__(self):
        return len(self.scene_views)

    def __getitem__(self, idx):
        return self.scene_views[idx]