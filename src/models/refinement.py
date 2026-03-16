import torch
import torch.nn as nn
from diffusers import UNet2DModel, DDPMScheduler

class DiffusionSegmentationRefiner(nn.Module):
    """
    Diffusion-Based Mask Refinement.
    Takes a weak pseudo-mask (generated from Grad-CAM) and refines it using a 
    lightweight diffusion model. It eliminates the need for dense pixel-level labels.
    """
    def __init__(self, image_size=224, in_channels=4, model_channels=64, num_train_timesteps=1000):
        """
        in_channels = 3 (RGB Image) + 1 (Grad-CAM Coarse Mask) = 4
        """
        super(DiffusionSegmentationRefiner, self).__init__()
        
        self.unet = UNet2DModel(
            sample_size=image_size,
            in_channels=in_channels,    # Image + Mask Prior
            out_channels=1,             # Output probability map for grain mask
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
        
        # the scheduler dictates the noise schedule (forward and reverse)
        self.scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)
        
    def forward(self, noisy_mask, t, image_cond):
        """
        Forward pass to predict noise during training.
        noisy_mask: x_t (Noisy ground truth or prediction) [B, 1, H, W]
        t: Time step
        image_cond: Original RGB + GradCAM heatmap concatenated [B, 4, H, W]
        """
        # The UNet takes in the noisy target mask and conditioning input merged together.
        # But this standard UNet2DModel doesn't have an explicit 'condition' argument 
        # unless we modify it or stack them. Here we stack them along channel dim. 
        # But image_cond already has 4 channels. It doesn't quite match to just stack mask+image.
        # Let's assume the unet receives (noisy mask + RGB conditional).
        # We need unet in_channels = 1 + 3 = 4.
        
        model_input = torch.cat([noisy_mask, image_cond], dim=1) 
        # If image_cond is RGB (3 channels) and noisy_mask is 1 channel, totality = 4 channels.
        
        # Predict the noise residual
        noise_pred = self.unet(model_input, t).sample
        return noise_pred
    
    @torch.no_grad()
    def refine_mask(self, image, grad_cam_mask, num_inference_steps=50):
        """
        Inference routine.
        Starting from pure noise, iteratively denoises conditioned on the original 
        image and the coarse Grad-CAM mask to produce a crisp boundary mask.
        """
        self.scheduler.set_timesteps(num_inference_steps)
        batch_size = image.shape[0]
        
        # 1. Start with Gaussian Noise
        x = torch.randn((batch_size, 1, image.shape[2], image.shape[3]), device=image.device)
        
        # 2. Condition on RGB Image (assuming image has 3 channels)
        # Note: Depending on the specific diffusion framing, some researchers condition on just the RGB image,
        # others condition on RGB + GradCAM. Let's assume conditioning = RGB.
        condition = image # [B, 3, H, W]
        
        # 3. Denoising loop
        for t in self.scheduler.timesteps:
            # Predict noise
            model_input = torch.cat([x, condition], dim=1) # [B, 4, H, W]
            noise_pred = self.unet(model_input, t).sample
            
            # Step the diffusion process
            x = self.scheduler.step(noise_pred, t, x).prev_sample
            
        # 4. Final output is un-normalized or loosely bounded, push through sigmoid
        refined_mask = torch.sigmoid(x)
        return refined_mask
