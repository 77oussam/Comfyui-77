import torch
import numpy as np
from PIL import Image

class HueSaturationLightness77:
    """
    A ComfyUI node for adjusting Hue, Saturation, and Lightness,
    similar to Photoshop's HSL adjustment layer.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "hue_shift": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "lightness": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "adjust_hsl"
    CATEGORY = "ALL-77"

    def tensor_to_np(self, tensor):
        # Converts a torch tensor (B, H, W, C) to a NumPy array (B, H, W, C)
        return (tensor.cpu().numpy() * 255).astype(np.uint8)

    def np_to_tensor(self, np_array):
        # Converts a NumPy array (B, H, W, C) to a torch tensor (B, H, W, C)
        return torch.from_numpy(np_array.astype(np.float32) / 255.0)

    def rgb_to_hsl(self, rgb):
        # Vectorized RGB to HSL conversion
        rgb = rgb / 255.0
        max_c = np.max(rgb, axis=-1)
        min_c = np.min(rgb, axis=-1)
        
        l = (max_c + min_c) / 2.0
        
        delta = max_c - min_c
        h = np.zeros_like(l)
        s = np.zeros_like(l)
        
        non_zero_delta = delta != 0
        s[non_zero_delta] = delta[non_zero_delta] / (1 - np.abs(2 * l[non_zero_delta] - 1))
        
        idx_r = (max_c == rgb[..., 0]) & non_zero_delta
        idx_g = (max_c == rgb[..., 1]) & non_zero_delta
        idx_b = (max_c == rgb[..., 2]) & non_zero_delta
        
        h[idx_r] = ((rgb[idx_r, 1] - rgb[idx_r, 2]) / delta[idx_r]) % 6
        h[idx_g] = ((rgb[idx_g, 2] - rgb[idx_g, 0]) / delta[idx_g]) + 2
        h[idx_b] = ((rgb[idx_b, 0] - rgb[idx_b, 1]) / delta[idx_b]) + 4
        
        h = h / 6.0
        
        return np.stack([h, s, l], axis=-1)

    def hsl_to_rgb(self, hsl):
        # Vectorized HSL to RGB conversion
        h, s, l = hsl[..., 0], hsl[..., 1], hsl[..., 2]
        
        c = (1 - np.abs(2 * l - 1)) * s
        x = c * (1 - np.abs((h * 6) % 2 - 1))
        m = l - c / 2.0
        
        rgb = np.zeros(hsl.shape)
        
        # Conditions for different hue ranges
        idx = (h < 1/6)
        rgb[idx] = np.stack([c[idx], x[idx], np.zeros_like(c[idx])], axis=-1)
        idx = (h >= 1/6) & (h < 2/6)
        rgb[idx] = np.stack([x[idx], c[idx], np.zeros_like(c[idx])], axis=-1)
        idx = (h >= 2/6) & (h < 3/6)
        rgb[idx] = np.stack([np.zeros_like(c[idx]), c[idx], x[idx]], axis=-1)
        idx = (h >= 3/6) & (h < 4/6)
        rgb[idx] = np.stack([np.zeros_like(c[idx]), x[idx], c[idx]], axis=-1)
        idx = (h >= 4/6) & (h < 5/6)
        rgb[idx] = np.stack([x[idx], np.zeros_like(c[idx]), c[idx]], axis=-1)
        idx = (h >= 5/6)
        rgb[idx] = np.stack([c[idx], np.zeros_like(c[idx]), x[idx]], axis=-1)
        
        rgb = (rgb + m[..., np.newaxis]) * 255.0
        return np.clip(rgb, 0, 255).astype(np.uint8)

    def adjust_hsl(self, image, hue_shift, saturation, lightness):
        img_np = self.tensor_to_np(image)
        
        # Convert to HSL
        hsl_np = self.rgb_to_hsl(img_np)
        
        # Adjust Hue
        # Scale hue_shift from [-180, 180] to [0, 1] range
        hsl_np[..., 0] = (hsl_np[..., 0] + hue_shift / 360.0) % 1.0
        
        # Adjust Saturation
        hsl_np[..., 1] *= saturation
        
        # Adjust Lightness
        hsl_np[..., 2] += lightness
        
        # Clamp values
        np.clip(hsl_np[..., 1], 0.0, 1.0, out=hsl_np[..., 1])
        np.clip(hsl_np[..., 2], 0.0, 1.0, out=hsl_np[..., 2])
        
        # Convert back to RGB
        rgb_np = self.hsl_to_rgb(hsl_np)
        
        return (self.np_to_tensor(rgb_np),)

NODE_CLASS_MAPPINGS = {
    "HueSaturationLightness77": HueSaturationLightness77
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HueSaturationLightness77": "🎨 Hue/Saturation/Lightness 77"
}
