import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf

from torchvision.ops import masks_to_boxes
from torchvision.transforms.functional import to_tensor

from utils import get_gaussians_params
from submodules.gaussian_editor.gaussiansplatting.utils.graphics_utils import fov2focal
from submodules.gaussian_editor.threestudio.utils.dpt import DPT
from submodules.gaussian_editor.threestudio.utils.misc import get_device
from submodules.gaussian_editor.gaussiansplatting.scene import Scene
from submodules.gaussian_editor.gaussiansplatting.scene.vanilla_gaussian_model import (
    GaussianModel as VanillaGaussianModel,
)
from submodules.gaussian_editor.gaussiansplatting.scene.gaussian_model import (
    GaussianModel,
)
from submodules.gaussian_editor.gaussiansplatting.gaussian_renderer import render
from submodules.gaussian_editor.threestudio.utils.transform import (
    rotate_gaussians,
    translate_gaussians,
    scale_gaussians,
    default_model_mtx,
)
from submodules.gaussian_editor.gaussiansplatting.scene.cameras import Camera


def get_object_bbox_and_mask(
    mask_path: str, mask_width: int, mask_height: int
) -> torch.Tensor:
    mask = Image.open(mask_path).convert("RGB").resize((mask_width, mask_height))
    mask = np.array(mask)
    mask = mask[:, :, 2] > 0
    mask = torch.from_numpy(mask)
    bbox = masks_to_boxes(mask[None])[0].to("cuda")
    return bbox, mask


def get_object_on_bg_arr(
    object_on_bg_img_path: str, img_width: int, img_height: int
) -> torch.Tensor:
    object_on_bg_img = (
        Image.open(object_on_bg_img_path).convert("RGB").resize((img_width, img_height))
    )
    return torch.tensor(np.array(object_on_bg_img)).float().permute(2, 0, 1)


def get_estimated_depth(depth_estimator: DPT, img_arr: torch.Tensor) -> torch.Tensor:
    return depth_estimator(img_arr.moveaxis(0, -1)[None, ...]).squeeze()


def get_object_center(bbox: torch.Tensor, cam: Camera):
    object_center = (bbox[:2] + bbox[2:]) / 2
    fx = fov2focal(cam.FoVx, cam.image_width)
    fy = fov2focal(cam.FoVy, cam.image_height)
    object_center = (
        object_center - torch.tensor([cam.image_width, cam.image_height]).to("cuda") / 2
    ) / torch.tensor([fx, fy]).to("cuda")
    return object_center


def get_bg_gaussians(gaussians_path: str) -> GaussianModel:
    gaussians = GaussianModel(
        sh_degree=0,
        anchor_weight_init_g0=1.0,
        anchor_weight_init=0.1,
        anchor_weight_multiplier=2,
    )
    gaussians.load_ply(gaussians_path)
    return gaussians


def get_object_gaussians(gaussians_path: str, sh_degree: float) -> VanillaGaussianModel:
    gaussians = VanillaGaussianModel(sh_degree)
    gaussians.load_ply(gaussians_path)
    gaussians._opacity.data = torch.ones_like(gaussians._opacity.data) * 99.99
    return gaussians


def transfrom_object(object_gaussians, cam, T_in_cam, real_scale):
    object_gaussians._xyz.data -= object_gaussians._xyz.data.mean(dim=0, keepdim=True)

    rotate_gaussians(object_gaussians, default_model_mtx.T)

    object_scale = (
        object_gaussians._xyz.data.max(dim=0)[0]
        - object_gaussians._xyz.data.min(dim=0)[0]
    )[:2]

    relative_scale = (real_scale / object_scale).mean()
    print(relative_scale)

    scale_gaussians(object_gaussians, relative_scale)

    object_gaussians._xyz.data += T_in_cam

    R = torch.from_numpy(cam.R).float().cuda()
    T = -R @ torch.from_numpy(cam.T).float().cuda()

    rotate_gaussians(object_gaussians, R)
    translate_gaussians(object_gaussians, T)


def get_object_tranforms_params(
    cam,
    depth_scale,
    rendered_depth,
    bg_estimated_depth,
    object_estimated_depth,
    object_center,
):
    # assuming rendered_depth = a * estimated_depth + b
    y = rendered_depth
    x = bg_estimated_depth
    a = (torch.sum(x * y) - torch.sum(x) * torch.sum(y)) / (
        torch.sum(x**2) - torch.sum(x) ** 2
    )
    b = torch.sum(y) - a * torch.sum(x)

    z_in_cam = object_estimated_depth.min() * a + b
    scaled_z_in_cam = z_in_cam * depth_scale
    x_in_cam, y_in_cam = (object_center.cuda()) * scaled_z_in_cam
    T_in_cam = torch.stack([x_in_cam, y_in_cam, scaled_z_in_cam], dim=-1)

    bbox = bbox.cuda()
    fx = fov2focal(cam.FoVx, cam.image_width)
    fy = fov2focal(cam.FoVy, cam.image_height)
    real_scale = (
        (bbox[2:] - bbox[:2]) / torch.tensor([fx, fy], device="cuda") * z_in_cam
    )

    return T_in_cam, real_scale


def get_depths(
    bg_gaussians,
    cam,
    pipeline_params,
    object_mask,
    depth_estimator,
    object_on_bg_img_arr,
    bg_color,
):
    render_pkg = render(cam, bg_gaussians, pipeline_params, bg_color)
    rendered_depth = render_pkg["depth_3dgs"][..., ~object_mask]

    estimated_depth = get_estimated_depth(depth_estimator, object_on_bg_img_arr)
    bg_estimated_depth = estimated_depth[~object_mask]
    object_estimated_depth = estimated_depth[..., object_mask]

    min_object_estimated_depth = torch.quantile(object_estimated_depth, 0.05)
    max_object_estimated_depth = torch.quantile(object_estimated_depth, 0.95)
    obj_depth_scale = (max_object_estimated_depth - min_object_estimated_depth) * 2

    min_valid_depth_mask = (
        min_object_estimated_depth - obj_depth_scale
    ) < bg_estimated_depth
    max_valid_depth_mask = bg_estimated_depth < (
        max_object_estimated_depth + obj_depth_scale
    )
    valid_depth_mask = torch.logical_and(min_valid_depth_mask, max_valid_depth_mask)

    rendered_depth = rendered_depth[0, valid_depth_mask]
    bg_estimated_depth = bg_estimated_depth[valid_depth_mask.squeeze()]
    return rendered_depth, bg_estimated_depth, object_estimated_depth


def place_object_in_bg(
    bg_gaussians: GaussianModel,
    object_gaussians: VanillaGaussianModel,
    cam: Camera,
    depth_scale: float,
    pipeline_params,
    training_params,
    object_mask: torch.Tensor,
    object_bbox: torch.Tensor,
    depth_estimator: DPT,
    object_on_bg_arr: torch.Tensor,
    bg_color: torch.Tensor,
):
    rendered_depth, bg_estimated_depth, object_estimated_depth = get_depths(
        bg_gaussians,
        cam,
        pipeline_params,
        object_mask,
        depth_estimator,
        object_on_bg_arr,
        bg_color,
    )
    object_center = get_object_center(object_bbox, cam)
    T_in_cam, real_scale = get_object_tranforms_params(
        cam,
        depth_scale,
        rendered_depth,
        bg_estimated_depth,
        object_estimated_depth,
        object_center,
    )
    transfrom_object(object_gaussians, cam, T_in_cam, real_scale)
    bg_gaussians.training_setup(training_params)
    bg_gaussians.concat_gaussians(object_gaussians)


def get_config(base_config_file_path: str) -> dict:
    base_configs = OmegaConf.load(base_config_file_path)
    cli_configs = OmegaConf.from_cli()
    config = OmegaConf.merge(base_configs, cli_configs)
    return OmegaConf.to_container(config, resolve=True)


def get_cams(bg_gaussians, bg_dataset_params, iteration, main_cam_id, second_cam_id):
    scene = Scene(
        bg_dataset_params, bg_gaussians, load_iteration=iteration, shuffle=False
    )
    all_cams = scene.getTrainCameras()
    main_cam = all_cams[main_cam_id]
    second_cam = all_cams[second_cam_id]
    return main_cam, second_cam


def get_view_score(rendered_img):
    pass


def main():
    config = get_config("src/config.yml")

    bg_gaussians = get_bg_gaussians(config["bg_gaussians_path"])
    bg_gaussians_params = get_gaussians_params(config["bg_scene_path"])
    object_gaussians = get_object_gaussians(
        config["object_gaussians_path"],
        bg_gaussians_params["dataset_params"].sh_degree,
    )
    main_cam, second_cam = get_cams(
        bg_gaussians,
        bg_gaussians_params["dataset_params"],
        bg_gaussians_params["iteration"],
        config["cams"]["main_cam_id"],
        config["cams"]["second_cam_id"],
    )

    object_bbox, object_mask = get_object_bbox_and_mask(
        config["object_mask_path"],
        config["resolution"]["width"],
        config["resolution"]["height"],
    )
    depth_estimator = DPT(get_device(), mode="depth")
    object_on_bg_arr = get_object_on_bg_arr(
        config["object_on_bg_img_path"],
        config["resolution"]["width"],
        config["resolution"]["height"],
    )

    render_bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

    for depth_scale in range(
        config["depth_scale"]["start"],
        config["depth_scale"]["end"],
        config["depth_scale"]["step"],
    ):
        place_object_in_bg(
            bg_gaussians,
            object_gaussians,
            main_cam,
            depth_scale,
            bg_gaussians_params["pipeline_params"],
            bg_gaussians_params["training_params"],
            object_mask,
            object_bbox,
            depth_estimator,
            object_on_bg_arr,
        )

        second_view_render_pkg = render(
            second_cam,
            bg_gaussians,
            bg_gaussians_params["pipeline_params"],
            render_bg_color,
        )
        score = get_view_score(second_view_render_pkg["render"])


if __name__ == "__main__":
    main()
