import torch
import torch.nn as nn
from diffusers import UNet2DModel, DDPMScheduler

class DiffusionSegmentationRefiner(nn.Module):
    """Diffusion-based mask refinement."""
    def __init__(self, image_size=224, in_channels=4, model_channels=64, num_train_timesteps=1000):
        super(DiffusionSegmentationRefiner, self).__init__()
        
        self.unet = UNet2DModel(
            sample_size=image_size,
            in_channels=in_channels,
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(model_channels, model_channels*2, model_channels*4, model_channels*8),
            down_block_types=(
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
            ),
        )
        
        self.scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)
        
    def forward(self, noisy_mask, t, image_cond):
        """Predict noise during training."""
        # in_channels = 1 (mask) + 3 (RGB)
        model_input = torch.cat([noisy_mask, image_cond], dim=1) 
        noise_pred = self.unet(model_input, t).sample
        return noise_pred
    
    @torch.no_grad()
    def refine_mask(self, image, grad_cam_mask, num_inference_steps=50):
        """Denoise to produce refined mask."""
        self.scheduler.set_timesteps(num_inference_steps)
        batch_size = image.shape[0]
        
        x = torch.randn((batch_size, 1, image.shape[2], image.shape[3]), device=image.device)
        condition = image

        for t in self.scheduler.timesteps:
            model_input = torch.cat([x, condition], dim=1)
            noise_pred = self.unet(model_input, t).sample
            x = self.scheduler.step(noise_pred, t, x).prev_sample
            
        refined_mask = torch.sigmoid(x)
        return refined_mask
