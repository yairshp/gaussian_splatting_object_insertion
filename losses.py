import torch
from torch.nn.functional import mse_loss
from diffusers import AutoencoderKL, StableDiffusionImg2ImgPipeline
from torchvision import transforms

# vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="vae", revision=None).to('cuda')
# diffusion_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)
diffusion_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
diffusion_pipeline.enable_sequential_cpu_offload()

def vae_reconstrucion_loss(img):
    encoding_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    img = transforms.ToPILImage()(img).convert('RGB').resize((512, 512))
    img = encoding_transforms(img).to('cuda')
    latent = vae(img.to(torch.float32))
    # latent = vae.encode(img)
    reconstructed = vae.decode(latent)
    return mse_loss(reconstructed, img)

def diffusion_reconstruction_loss(img):
    reconstructed = diffusion_pipeline(prompt='', image=img, strength=0.5, output_type='pt').images[0]
    padded = torch.nn.functional.pad(reconstructed, (0,0,1,3), "constant", 0)
    return mse_loss(padded, img)